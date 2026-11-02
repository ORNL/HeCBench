// One work-group per matrix, one launch. Local memory holds the lower
// Cholesky factor of I+A and then its inverse, plus a 64x64 scratch for the
// block couplings when the device has room. The inverse is written to global
// T, B is loaded over the factor, and J = B * inv is written to global
// memory. T is then scaled by alpha.
//
// Wave is the device sub-group size, 32 or 64. Each lane owns one column of
// a Wave x Wave diagonal tile. Coupling then joins those tiles
// up to 128 with the block formula inv([A 0; C B]) = [invA 0; -invB*C*invA, invB].
// local_layout.hpp picks ld 129 with the scratch on chip at 82432 bytes, or
// ld 128 with that scratch in global memory at 64KB.

#include "fused.h"
#include "local_layout.hpp"

#include <algorithm>
#include <cstdio>

namespace {

constexpr int kDim = 128;
constexpr int kThreads = 256;
constexpr size_t kTightFloats = kVdiTightBytes / sizeof(float);
constexpr size_t kPaddedFloats = kVdiPaddedBytes / sizeof(float);

int g_wave = 0;
int g_nmat = 0;
int g_scratch_local = 0;
size_t g_device_max = 0;
float* g_scratch = nullptr;
int* g_pivot = nullptr;
char g_desc[192] = "local memory not queried";

using Item = sycl::nd_item<1>;
// Decorated local pointer. A raw float* is a global parameter, so decaying
// the local accessor to float* made every factor load miss local memory.
using Local = sycl::multi_ptr<float, sycl::access::address_space::local_space,
                              sycl::access::decorated::yes>;

// Same right-looking factorization as the CUDA kernel: two work-items per
// row, each applying a 4-wide update to half of the columns (k, row].
void chol_lower(Local L, int ld, int n, int tid, const sycl::group<1>& grp, int* pivot_fail) {
  const int row = tid & (n - 1);
  const int part = tid >> 7;
  auto rp = L + row * ld;
  for (int k = 0; k < n; ++k) {
    if (tid == k) {
      const float d = L[k * ld + k];
      if (!(d > 0.0f)) {
        sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            flag(*pivot_fail);
        flag.store(1);
      }
      L[k * ld + k] = sycl::sqrt(d);
    }
    sycl::group_barrier(grp);
    const float invd = 1.0f / L[k * ld + k];
    if (part == 0 && row > k) rp[k] *= invd;
    sycl::group_barrier(grp);
    if (row > k) {
      const float lik = rp[k];
      const int lo = k + 1;
      const int hi = row;
      const int mid = (lo + hi) >> 1;
      const int j0 = part ? mid + 1 : lo;
      const int j1 = part ? hi : mid;
      int j = j0;
      #pragma unroll 1
      for (; j + 3 <= j1; j += 4) {
        const float b0 = L[(j + 0) * ld + k];
        const float b1 = L[(j + 1) * ld + k];
        const float b2 = L[(j + 2) * ld + k];
        const float b3 = L[(j + 3) * ld + k];
        rp[j] -= lik * b0;
        rp[j + 1] -= lik * b1;
        rp[j + 2] -= lik * b2;
        rp[j + 3] -= lik * b3;
      }
      #pragma unroll 1
      for (; j <= j1; ++j) rp[j] -= lik * L[j * ld + k];
    }
    sycl::group_barrier(grp);
  }
}

// Same inverse as CUDA warp_trinv32, for this binary's sub-group. Lane `lane`
// owns column `lane`: it loads that column into registers, solves L x = e_lane,
// and writes the column back. The sum for row i only includes columns j with
// lane <= j < i. L[i, j] is broadcast from the lane that owns column j, so the
// solve does not reread local memory. No barrier inside the tile: every read
// of the factor finished before any column is stored.
template <int Wave>
void wave_trinv(Local S, int ld, int r0, int lane, sycl::sub_group &sg) {
  float colv[Wave];
  for (int i = 0; i < Wave; ++i)
    colv[i] = S[(r0 + i) * ld + (r0 + lane)];

  float x[Wave];
  for (int i = 0; i < Wave; ++i) x[i] = 0.0f;
  x[lane] = 1.0f / colv[lane];

  for (int i = 1; i < Wave; ++i) {
    float sum = 0.0f;
    for (int j = 0; j < i; ++j) {
      const float Lij = sycl::group_broadcast(sg, colv[i], sycl::id<1>(j));
      if (j >= lane) sum += Lij * x[j];
    }
    const float Lii = sycl::group_broadcast(sg, colv[i], sycl::id<1>(i));
    if (i > lane) x[i] = -sum / Lii;
  }

  for (int i = lane; i < Wave; ++i)
    S[(r0 + i) * ld + (r0 + lane)] = x[i];
}

template <typename PtrA, typename PtrB, typename PtrC>
void gemm_nn(PtrA A, int lda, PtrB B, int ldb, PtrC C, int ldc,
             int M, int N, int K, int tid, float scale) {
  constexpr int W = 8;
  const int chunks = N / W;
  const int ntasks = M * chunks;
  for (int task = tid; task < ntasks; task += kThreads) {
    const int row = task / chunks;
    const int c0 = (task - row * chunks) * W;
    float a[W] = {};
    auto arow = A + row * lda;
    for (int k = 0; k < K; ++k) {
      const float aik = arow[k];
      auto bk = B + k * ldb + c0;
      #pragma unroll
      for (int u = 0; u < W; ++u) a[u] += aik * bk[u];
    }
    auto crow = C + row * ldc + c0;
    #pragma unroll
    for (int u = 0; u < W; ++u) crow[u] = scale * a[u];
  }
}

template <int Wave>
void gemm_tn_full(Local X, int ldx, float* C, int ldc, int n, int lane, int wave) {
  constexpr int W = 8;
  constexpr int kRows = kDim / Wave;
  const int nwaves = kThreads / Wave;
  const int panels = n / W;
  for (int p = wave; p < panels; p += nwaves) {
    const int c0 = p * W;
    float acc[kRows][W];
    #pragma unroll
    for (int i = 0; i < kRows; ++i)
      #pragma unroll
      for (int u = 0; u < W; ++u) acc[i][u] = 0.f;
    for (int k = 0; k < n; ++k) {
      float b[W];
      #pragma unroll
      for (int u = 0; u < W; ++u) b[u] = X[k * ldx + c0 + u];
      #pragma unroll
      for (int i = 0; i < kRows; ++i) {
        const float x = X[k * ldx + lane + i * Wave];
        #pragma unroll
        for (int u = 0; u < W; ++u) acc[i][u] += x * b[u];
      }
    }
    #pragma unroll
    for (int i = 0; i < kRows; ++i) {
      float* crow = C + (lane + i * Wave) * ldc + c0;
      #pragma unroll
      for (int u = 0; u < W; ++u) crow[u] = acc[i][u];
    }
  }
}

template <int Wave>
void gemm_row_panel(Local B, int ldb, const float* Inv, int ldi, float* J,
                    int n, int lane, int wave) {
  constexpr int W = 8;
  constexpr int kRows = kDim / Wave;
  const int nwaves = kThreads / Wave;
  const int panels = n / W;
  for (int p = wave; p < panels; p += nwaves) {
    const int c0 = p * W;
    float acc[kRows][W];
    float part[kRows][W];
    #pragma unroll
    for (int i = 0; i < kRows; ++i)
      #pragma unroll
      for (int u = 0; u < W; ++u) acc[i][u] = part[i][u] = 0.f;
    for (int k = 0; k < n; ++k) {
      float b[W];
      #pragma unroll
      for (int u = 0; u < W; ++u) b[u] = Inv[k * ldi + c0 + u];
      #pragma unroll
      for (int i = 0; i < kRows; ++i) {
        const float x = B[(lane + i * Wave) * ldb + k];
        #pragma unroll
        for (int u = 0; u < W; ++u) part[i][u] += x * b[u];
      }
      if ((k & 7) == 7) {
        #pragma unroll
        for (int i = 0; i < kRows; ++i)
          #pragma unroll
          for (int u = 0; u < W; ++u) {
            acc[i][u] += part[i][u];
            part[i][u] = 0.f;
          }
      }
    }
    #pragma unroll
    for (int i = 0; i < kRows; ++i) {
      float* out = J + (lane + i * Wave) * n + c0;
      #pragma unroll
      for (int u = 0; u < W; ++u) out[u] = acc[i][u];
    }
  }
}

template <int Wave, bool ScratchLocal>
void vdn_delta_factors(Item it, Local L, const float* A, const float* B, const float* alpha,
                 float* T, float* J, float* scratch, int nmat, int* pivot_fail) {
  constexpr int N = kDim;
  constexpr int LD = ScratchLocal ? 129 : 128;
  const sycl::group<1> grp = it.get_group();
  const int m = grp.get_group_id(0);
  if (m >= nmat) return;
  const int tid = it.get_local_linear_id();
  const auto sg = it.get_sub_group();
  const int lane = sg.get_local_linear_id();
  const int wave = sg.get_group_linear_id();
  // On-chip when the device has at least 82432 bytes. Otherwise one 64x64
  // global scratch per matrix, the same buffer HIP uses.
  float* ScratchG = nullptr;
  if constexpr (!ScratchLocal)
    ScratchG = scratch + static_cast<size_t>(m) * 64 * 64;

  const float* Am = A + static_cast<size_t>(m) * N * N;
  const float* Bm = B + static_cast<size_t>(m) * N * N;
  float* Tm = T + static_cast<size_t>(m) * N * N;
  float* Jm = J + static_cast<size_t>(m) * N * N;
  const float* arow = alpha + static_cast<size_t>(m) * N;

  const sycl::float4* A4 = reinterpret_cast<const sycl::float4*>(Am);
  for (int idx = tid; idx < (N * N) / 4; idx += kThreads) {
    const sycl::float4 v = A4[idx];
    const int base = idx * 4;
    const int r = base >> 7;
    const int c = base & 127;
    float vals[4] = {v[0], v[1], v[2], v[3]};
    for (int u = 0; u < 4; ++u) {
      const int cc = c + u;
      float x = vals[u];
      if (r == cc) x += 1.0f;
      L[r * LD + cc] = (r >= cc) ? x : 0.0f;
    }
  }
  // Work-group scope, the same fence as a CUDA/HIP block barrier. A device
  // scope fence stalls every queue on the GPU at each of the 128 pivots.
  sycl::group_barrier(grp);

  chol_lower(L, LD, N, tid, grp, pivot_fail);
  sycl::group_barrier(grp);

  if (wave < N / Wave) wave_trinv<Wave>(L, LD, wave * Wave, lane, sg);
  sycl::group_barrier(grp);

  for (int blk = Wave; blk < N; blk *= 2) {
    const int npairs = N / (2 * blk);
    for (int p = 0; p < npairs; ++p) {
      const int r0 = p * 2 * blk;
      if constexpr (ScratchLocal) {
        auto Scratch = L + LD * N;
        gemm_nn(L + (r0 + blk) * LD + r0, LD, L + r0 * LD + r0, LD, Scratch, blk,
                blk, blk, blk, tid, 1.0f);
        sycl::group_barrier(grp);
        gemm_nn(L + (r0 + blk) * LD + (r0 + blk), LD, Scratch, blk,
                L + (r0 + blk) * LD + r0, LD, blk, blk, blk, tid, -1.0f);
      } else {
        gemm_nn(L + (r0 + blk) * LD + r0, LD, L + r0 * LD + r0, LD, ScratchG, blk,
                blk, blk, blk, tid, 1.0f);
        sycl::group_barrier(grp);
        gemm_nn(L + (r0 + blk) * LD + (r0 + blk), LD, ScratchG, blk,
                L + (r0 + blk) * LD + r0, LD, blk, blk, blk, tid, -1.0f);
      }
      sycl::group_barrier(grp);
    }
  }

  gemm_tn_full<Wave>(L, LD, Tm, N, N, lane, wave);
  sycl::group_barrier(grp);

  const sycl::float4* B4 = reinterpret_cast<const sycl::float4*>(Bm);
  for (int idx = tid; idx < (N * N) / 4; idx += kThreads) {
    const sycl::float4 v = B4[idx];
    const int base = idx * 4;
    const int r = base >> 7;
    const int c = base & 127;
    L[r * LD + c] = v[0];
    L[r * LD + c + 1] = v[1];
    L[r * LD + c + 2] = v[2];
    L[r * LD + c + 3] = v[3];
  }
  sycl::group_barrier(grp);
  gemm_row_panel<Wave>(L, LD, Tm, N, Jm, N, lane, wave);
  sycl::group_barrier(grp);
  // B has been consumed. Reuse the dead factor buffer for alpha so the row
  // scale is a local-memory broadcast without a separate allocation.
  if (tid < N) L[tid] = arow[tid];
  sycl::group_barrier(grp);
  for (int idx = tid; idx < N * N; idx += kThreads) {
    const int r = idx >> 7;
    Tm[idx] *= L[r];
  }
}

template <int Wave, bool ScratchLocal>
void submit(sycl::queue& q, const float* A, const float* B, const float* alpha,
            float* T, float* J, float* scratch, int* pivot_fail, int nmat) {
  constexpr size_t nfloat = ScratchLocal ? kPaddedFloats : kTightFloats;
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> smem(sycl::range<1>(nfloat), h);
    // 256 work-items. The caller picks Wave from the device sub-group size.
    h.parallel_for(sycl::nd_range<1>(static_cast<size_t>(nmat) * kThreads, kThreads),
                   [=](Item it) {
                     Local L = smem.template get_multi_ptr<sycl::access::decorated::yes>();
                     vdn_delta_factors<Wave, ScratchLocal>(it, L, A, B, alpha, T, J, scratch,
                                                    nmat, pivot_fail);
                   });
  });
}

}  // namespace

int fused_wave() { return g_wave; }

const char* fused_layout() { return g_desc; }

int fused_prepare(const sycl::device& dev) {
  auto sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
  auto r = std::max_element(sg_sizes.begin(), sg_sizes.end());
  g_wave = *r;
  if (g_wave != 64 && g_wave != 32) {
    fprintf(stderr, "fused: device sub-group is not 32 or 64\n");
    return 1;
  }
  // Device ceiling. The CUDA unified-runtime adapter reports the opt-in maximum
  // and sets the dynamic shared-memory attribute on launch. The HIP adapter
  // reports MaxSharedMemoryPerBlock and does not opt in:
  // https://github.com/intel/llvm/issues/23365
  g_device_max = dev.get_info<sycl::info::device::local_mem_size>();
  VdiLocalLayout lay{};
  if (!vdi_bind_local(g_device_max, lay)) return 1;
  g_scratch_local = lay.scratch_local;
  vdi_format_layout(g_desc, sizeof(g_desc), lay, g_device_max);
  return 0;
}

int fused_init(sycl::queue& q, int nmat) {
  if (nmat < 1) return 1;
  fused_shutdown(q);
  g_scratch = nullptr;
  // The padded layout keeps the 64x64 scratch in local memory. The 64KB
  // layout needs one global 64x64 per matrix
  if (!g_scratch_local) {
    g_scratch = sycl::malloc_device<float>(static_cast<size_t>(nmat) * 64 * 64, q);
    if (!g_scratch) {
      fprintf(stderr, "fused: scratch alloc failed\n");
      return 1;
    }
  }
  g_pivot = sycl::malloc_device<int>(1, q);
  if (!g_pivot) {
    fprintf(stderr, "fused: pivot flag alloc failed\n");
    fused_shutdown(q);
    return 1;
  }
  q.memset(g_pivot, 0, sizeof(int));
  q.wait_and_throw();
  g_nmat = nmat;
  return 0;
}

void fused_shutdown(sycl::queue& q) {
  if (g_scratch) sycl::free(g_scratch, q);
  if (g_pivot) sycl::free(g_pivot, q);
  g_scratch = nullptr;
  g_pivot = nullptr;
  g_nmat = 0;
}

int fused_pivot_failed(sycl::queue& q) {
  int host = 1;
  if (!g_pivot) return 1;
  q.memcpy(&host, g_pivot, sizeof(int));
  q.wait_and_throw();
  return host;
}

int fused_launch(sycl::queue& q, const float* A, const float* B, const float* alpha,
                 float* T, float* J, int nmat) {
  if (nmat != g_nmat || !g_pivot || (g_wave != 32 && g_wave != 64) ||
      (!g_scratch_local && !g_scratch)) {
    fprintf(stderr, "fused launch failed\n");
    return 1;
  }
  if (g_wave == 64) {
    if (g_scratch_local) submit<64, true>(q, A, B, alpha, T, J, g_scratch, g_pivot, nmat);
    else submit<64, false>(q, A, B, alpha, T, J, g_scratch, g_pivot, nmat);
  } else {
    if (g_scratch_local) submit<32, true>(q, A, B, alpha, T, J, g_scratch, g_pivot, nmat);
    else submit<32, false>(q, A, B, alpha, T, J, g_scratch, g_pivot, nmat);
  }
  return 0;
}
