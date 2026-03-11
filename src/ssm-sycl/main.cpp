// Modified codes from Claude Free

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <sycl/sycl.hpp>
#include "block_scan.hpp"
#include "reference.h"

static constexpr int kBatch  = 32;
static constexpr int kDim    = 2048; // D - feature / model dimension
static constexpr int kDstate =  8;   // N - SSM state size
static constexpr int kSeqLen = 1024; // L - sequence length (must be divisible
                                     //     by kNThreads*kNItems for even-len path)

inline float softplus_f(float x) {
    return (x > 20.f) ? x : sycl::native::log(1.f + sycl::native::exp(x));
}
inline float silu_f(float x) {
    return x / (1.f + sycl::native::exp(-x));
}

void selective_scan_base(
    const float* __restrict__ u,
    const float* __restrict__ delta,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ C,
    const float* __restrict__ D_vec,
    const float* __restrict__ delta_bias,
    const float* __restrict__ z,         // may be nullptr
    bool   delta_softplus,
    int    batch, int dim, int dstate, int seqlen,
    float* __restrict__ ssm_states,
    float* __restrict__ y,
    sycl::nd_item<3> &item
) {
    int b = item.get_group(2);
    int d = item.get_group(1) * item.get_local_range(2) + item.get_local_id(2);
    if (b >= batch || d >= dim) return;

    float db = delta_bias[d];
    float Dv = D_vec[d];
    float h[kDstate];
    for (int n = 0; n < dstate; ++n)
       h[n] = ssm_states[b*dim*dstate + d*dstate + n];
    
    for (int l = 0; l < seqlen; ++l) {
        float u_v  = u    [b*dim*seqlen + d*seqlen + l];
        float dt   = delta[b*dim*seqlen + d*seqlen + l] + db;
        if (delta_softplus) dt = softplus_f(dt);

        float yv = Dv * u_v;
        for (int n = 0; n < dstate; ++n) {
            float dA = sycl::native::exp(dt * A[d * dstate + n]);
            float dB = dt * B[b*dstate*seqlen + n*seqlen + l];
            h[n] = dA * h[n] + dB * u_v;
            yv  += C[b*dstate*seqlen + n*seqlen + l] * h[n];
        }
        if (z) yv *= silu_f(z[b*dim*seqlen + d*seqlen + l]);
        y[b*dim*seqlen + d*seqlen + l] = yv;
    }
    for (int n = 0; n < dstate; ++n)
      ssm_states[b*dim*dstate + d*dstate + n] = h[n];
}

template<int kNThreads_, int kNItems_, int kDstate_, bool kHasZ_,
         bool kDeltaSoftplus_, bool kIsEvenLen_>
struct SSMFwdKernelTraits {
    static constexpr int kNThreads      = kNThreads_;
    static constexpr int kNItems        = kNItems_;
    static constexpr int kDstate        = kDstate_;
    static constexpr bool kHasZ         = kHasZ_;
    static constexpr bool kDeltaSoftplus = kDeltaSoftplus_;
    static constexpr bool kIsEvenLen    = kIsEvenLen_;
    static constexpr int kChunkSize     = kNThreads * kNItems;

    // Associative scan element: (a, b) representing  h_t = a*h_{prev} + b
    struct ScanElement {
        float a;   // multiplicative factor (= A_bar)
        float b;   // additive   factor   (= B_bar * u)
    };

    // CUB associative operator  ∘
    struct ScanOp {
        inline ScanElement operator()(const ScanElement &lhs,
                                      const ScanElement &rhs) const {
            // rhs is "later" in time:  h = rhs.a*(lhs.a*h0 + lhs.b) + rhs.b
            return { rhs.a * lhs.a,  rhs.a * lhs.b + rhs.b };
        }
    };

    using BlockScanT = BlockScan<ScanElement, kNThreads, BLOCK_SCAN_WARP_SCANS>;

    // Shared memory:  one BlockScan workspace + tile buffers
    // running_prefix_smem[n]: broadcast slot so the thread that owns the last
    // valid sequence position within this chunk can share its carry with:
    //   (a) all other threads for y accumulation, and
    //   (b) threads 0..kDstate-1 for the final ssm_states writeback.
    // last_valid_thread: index of the thread that owns seq position
    //   min(chunk_end, seqlen) - 1  (needed for non-even-length chunks).
    struct SharedMemory {
        typename BlockScanT::TempStorage scan_storage;
        float u_tile    [kChunkSize];
        float delta_tile[kChunkSize];
        // B and C: (kDstate, kChunkSize) – laid out [state_dim][time]
        float B_tile    [kDstate * kChunkSize];
        float C_tile    [kDstate * kChunkSize];
        // Per-state carry broadcast slot (written by last-valid thread, read by all)
        float carry_a   [kDstate];
        float carry_b   [kDstate];
        int   last_valid_thread;  // thread index owning the last valid seq element
        int   last_valid_item;    // item index within that thread
    };

    static constexpr int kSmemSize = sizeof(SharedMemory);
};

struct SSMParamsBase {
    int    batch, dim, seqlen, dstate, n_chunks;
    // Pointers
    float *u_ptr, *delta_ptr, *A_ptr, *B_ptr, *C_ptr,
          *D_ptr, *delta_bias_ptr, *z_ptr,
          *ssm_states_ptr, *out_ptr;
    // Inter-chunk carry buffers (global memory, shape: batch*dim * n_chunks * dstate)
    // carry_a[chunk] and carry_b[chunk] store the ScanElement after each chunk.
    // chunk 0 initialises from ssm_states; chunks 1..n_chunks-1 read from here.
    float *running_prefix_a;   // multiplicative carry  (a field of ScanElement)
    float *running_prefix_b;   // additive    carry     (b field — the actual h)
    // Strides (elements, not bytes) — mirrors vLLM's explicit stride fields
    int u_batch_stride,      u_d_stride;        // for u and delta (same shape)
    int B_batch_stride,      B_dstate_stride;   // for B and C
    int states_batch_stride, states_d_stride;
    // Stride into the running_prefix buffers: [bd_idx * dstate * n_chunks + n * n_chunks + chunk]
    // (bd_idx = b*dim+d)
};

template <typename Traits>
void selective_scan_vllm_kernel(SSMParamsBase params,
                                uint8_t *smem_raw,
                                sycl::nd_item<3> &item)
{
    constexpr int kNThreads = Traits::kNThreads;
    constexpr int kNItems    = Traits::kNItems;
    constexpr int kChunkSize = Traits::kChunkSize;
    constexpr int kDstate    = Traits::kDstate;

    // Block → (batch, dim) mapping
    // Grid: (batch * dim,  n_chunks)
    const int bd_idx = item.get_group(2); // linearised batch*dim index
    const int chunk = item.get_group(1);  // which chunk of L

    const int b = bd_idx / params.dim;
    const int d = bd_idx % params.dim;

    const int chunk_start = chunk * kChunkSize;  // first seq index this block

    // Pointers to this row's data
    const float* u_row     = params.u_ptr
                             + b * params.u_batch_stride
                             + d * params.u_d_stride;
    const float* delta_row = params.delta_ptr
                             + b * params.u_batch_stride
                             + d * params.u_d_stride;
    const float* z_row     = (Traits::kHasZ) ?
                             params.z_ptr + b * params.u_batch_stride
                                          + d * params.u_d_stride
                             : nullptr;
    float*       out_row   = params.out_ptr
                             + b * params.u_batch_stride
                             + d * params.u_d_stride;

    auto& smem = *reinterpret_cast<typename Traits::SharedMemory*>(smem_raw);

    // Compute last valid thread/item within this chunk
    // "last valid" = the thread that processes seq position (last_seq_idx):
    //   even-len path: all positions in [chunk_start, chunk_start+CHUNK) valid
    //                  → last thread (kNThreads-1), last item (kNItems-1)
    //   odd-len  path: last valid seq idx = min(chunk_start+CHUNK, seqlen) - 1
    if (item.get_local_id(2) == 0) {
        if (Traits::kIsEvenLen) {
            smem.last_valid_thread = kNThreads - 1;
            smem.last_valid_item   = kNItems   - 1;
        } else {
            int last_seq =
                sycl::min(chunk_start + kChunkSize, params.seqlen) - 1;
            int local    = last_seq - chunk_start;          // 0-based within chunk
            smem.last_valid_thread = local / kNItems;
            smem.last_valid_item   = local % kNItems;
        }
    }
    item.barrier(sycl::access::fence_space::local_space);

    const int last_thr  = smem.last_valid_thread;
    const int last_item = smem.last_valid_item;

    // Load u and delta tiles into shared memory (coalesced)
    // Each thread loads kNItems consecutive elements.
    int base = chunk_start + item.get_local_id(2) * kNItems;
#pragma unroll
    for (int i = 0; i < kNItems; ++i) {
        int l = base + i;
        bool valid = Traits::kIsEvenLen || (l < params.seqlen);
        float u_v  = valid ? u_row    [l] : 0.f;
        float dt_v = valid ? delta_row[l] : 0.f;
        dt_v += params.delta_bias_ptr[d];
        if (Traits::kDeltaSoftplus) dt_v = softplus_f(dt_v);
        smem.u_tile[item.get_local_id(2) * kNItems + i] = u_v;
        smem.delta_tile[item.get_local_id(2) * kNItems + i] = dt_v;
    }
    item.barrier(sycl::access::fence_space::local_space);

    // Per-state-dimension parallel scan
    float D_val = params.D_ptr[d];

    // Initialise output accumulator to D * u (skip z for now)
    float y_tile[kNItems];
    #pragma unroll
    for (int i = 0; i < kNItems; ++i) {
        int li = item.get_local_id(2) * kNItems + i;
        y_tile[i] = D_val * smem.u_tile[li];
    }

    // Inter-chunk carry index layout:
    //   running_prefix_{a,b}[ bd_idx * kDstate * n_chunks + n * n_chunks + chunk ]
    // chunk==0: carry initialised from ssm_states (initial hidden state h0).
    // chunk>0 : carry loaded from the previous chunk's global-memory entry.
    // Within each state-dim iteration the carry is broadcast via smem.carry_{a,b}[n].

    // Iterate over each SSM state dimension independently
    for (int n = 0; n < kDstate; ++n) {

        // Load carry for this (bd_idx, n, chunk)
        // All threads need the same carry scalar; thread 0 loads it and
        // broadcasts via smem.carry_{a,b}[n].
        using SE = typename Traits::ScanElement;
        if (item.get_local_id(2) == 0) {
            SE carry;
            int carry_base = bd_idx * kDstate * params.n_chunks
                           + n      * params.n_chunks;
            if (chunk == 0) {
                // Encode initial state h0 as ScanElement{1, h0}:
                //   combined.b = prefix.a * 1 * h0 + prefix.b
                float h0 = params.ssm_states_ptr[b * params.states_batch_stride
                                                + d * params.states_d_stride + n];
                carry.a = 1.f;
                carry.b = h0;
            } else {
                carry.a = params.running_prefix_a[carry_base + (chunk - 1)];
                carry.b = params.running_prefix_b[carry_base + (chunk - 1)];
            }
            smem.carry_a[n] = carry.a;
            smem.carry_b[n] = carry.b;
        }

        // Load B and C tiles for state n
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) {
            int l = base + i;
            bool valid = Traits::kIsEvenLen || (l < params.seqlen);
            smem.B_tile[n * kChunkSize + item.get_local_id(2) * kNItems +
                        i] = valid
                                 ? params.B_ptr[b * params.B_batch_stride +
                                                n * params.B_dstate_stride + l]
                                 : 0.f;
            smem.C_tile[n * kChunkSize + item.get_local_id(2) * kNItems +
                        i] = valid
                                 ? params.C_ptr[b * params.B_batch_stride +
                                                n * params.B_dstate_stride + l]
                                 : 0.f;
        }
        item.barrier(sycl::access::fence_space::local_space); // carry_a/b[n] and B/C tiles are ready

        SE carry = { smem.carry_a[n], smem.carry_b[n] };

        // Build per-thread ScanElements (one per item)
        SE thread_data[kNItems];
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) {
            int li = item.get_local_id(2) * kNItems + i;
            float dt  = smem.delta_tile[li];
            float u_v = smem.u_tile[li];
            float Bn  = smem.B_tile[n * kChunkSize + li];
            float a_bar = sycl::native::exp(dt * params.A_ptr[d * kDstate + n]);
            float b_bar = dt * Bn * u_v;
            thread_data[i].a = a_bar;
            thread_data[i].b = b_bar;
        }

        // Inclusive prefix scan across the block
        using BlockScanT = typename Traits::BlockScanT;
        using ScanOp     = typename Traits::ScanOp;
        SE prefix_out[kNItems];
        BlockScanT(smem.scan_storage, item).InclusiveScan(thread_data, prefix_out, ScanOp{});
        item.barrier(sycl::access::fence_space::local_space);

        // Incorporate carry and accumulate y
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) {
            // combined = carry ∘ prefix_out[i]
            SE combined;
            combined.a = prefix_out[i].a * carry.a;
            combined.b = prefix_out[i].a * carry.b + prefix_out[i].b;
            float h_t = combined.b;
            int li = item.get_local_id(2) * kNItems + i;
            y_tile[i] += smem.C_tile[n * kChunkSize + li] * h_t;
        }

        // ── Write new carry to global memory and smem broadcast slot ─────────
        // The carry after this chunk = the combined ScanElement at the LAST
        // VALID sequence position within the chunk.
        // For even-length chunks this is always the last thread's last item.
        // For odd-length (padded) chunks this may be an earlier thread/item.
        // Using smem.last_valid_thread / last_valid_item computed above.
        if (item.get_local_id(2) == last_thr) {
            SE combined;
            combined.a = prefix_out[last_item].a * carry.a;
            combined.b = prefix_out[last_item].a * carry.b + prefix_out[last_item].b;

            // Write to global memory for subsequent chunks
            int carry_idx = bd_idx * kDstate * params.n_chunks
                          + n      * params.n_chunks
                          + chunk;
            params.running_prefix_a[carry_idx] = combined.a;
            params.running_prefix_b[carry_idx] = combined.b;

            // Also broadcast into smem so threads 0..kDstate-1 can read
            // it for the final state writeback (avoids a second global read).
            smem.carry_a[n] = combined.a;
            smem.carry_b[n] = combined.b;
        }
        item.barrier(sycl::access::fence_space::local_space);
    }
    // After the n-loop, smem.carry_{a,b}[n] holds the end-of-chunk hidden
    // state for every n, written by last_thr and visible to all threads.

    // Apply optional SiLU z-gate and write outputs
    #pragma unroll
    for (int i = 0; i < kNItems; ++i) {
        int l = base + i;
        bool valid = Traits::kIsEvenLen || (l < params.seqlen);
        if (!valid) continue;

        float yv = y_tile[i];
        if (Traits::kHasZ) yv *= silu_f(z_row[l]);
        out_row[l] = yv;
    }

    // Write final SSM state back to global memory (last chunk only)
    // Thread n writes state dimension n, reading from smem.carry_b[n] which
    // holds h_L = the hidden state after the last valid sequence position.
    if (chunk == params.n_chunks - 1 && item.get_local_id(2) < kDstate) {
        int n = item.get_local_id(2);
        params.ssm_states_ptr[b * params.states_batch_stride
                             + d * params.states_d_stride + n]
            = smem.carry_b[n];
    }
}

static constexpr int kNT = 128;
static constexpr int kNI = 4;

void launch_kernel(SSMParamsBase &p, bool has_z, bool softplus, bool even_len,
                   float *d_carry_a, float *d_carry_b,
                   sycl::queue &stream)
{
    p.running_prefix_a = d_carry_a;
    p.running_prefix_b = d_carry_b;
    // Instantiate all compile-time specialisations via nested lambdas,
    // mirroring vLLM's BOOL_SWITCH pattern.
    auto dispatch = [&]<bool HZ, bool SP, bool EL>() {
      using Traits = SSMFwdKernelTraits<kNT, kNI, kDstate, HZ, SP, EL>;
      int smem = Traits::kSmemSize;
      sycl::range<3> gws (1, p.n_chunks, p.batch * p.dim * kNT);
      sycl::range<3> lws (1, 1, kNT);
      stream.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> smem_acc(sycl::range<1>(smem), cgh);
        cgh.parallel_for(
          sycl::nd_range<3>(gws, lws), [=](sycl::nd_item<3> item) {
            selective_scan_vllm_kernel<Traits>(
              p,
              smem_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
              item);
          });
      });
    };

    // Dispatch (8 instantiations)
    if ( has_z &&  softplus &&  even_len) dispatch.operator()<true,  true,  true >();
    if ( has_z &&  softplus && !even_len) dispatch.operator()<true,  true,  false>();
    if ( has_z && !softplus &&  even_len) dispatch.operator()<true,  false, true >();
    if ( has_z && !softplus && !even_len) dispatch.operator()<true,  false, false>();
    if (!has_z &&  softplus &&  even_len) dispatch.operator()<false, true,  true >();
    if (!has_z &&  softplus && !even_len) dispatch.operator()<false, true,  false>();
    if (!has_z && !softplus &&  even_len) dispatch.operator()<false, false, true >();
    if (!has_z && !softplus && !even_len) dispatch.operator()<false, false, false>();
}

static void fill_rand(float* buf, int n, float lo, float hi) {
    for (int i = 0; i < n; ++i)
        buf[i] = lo + (hi - lo) * ((float)rand() / (float)RAND_MAX);
}
static float max_abs_err(const float* a, const float* b, int n) {
    float e = 0.f;
    for (int i = 0; i < n; ++i) e = fmaxf(e, fabsf(a[i] - b[i]));
    return e;
}
static void print_result(const char* label, float err_y, float err_s, float tol) {
    printf("  %-22s  err_y=%.3e  err_s=%.3e  %s\n",
           label, err_y, err_s,
           (err_y < tol && err_s < tol) ? "PASS" : "FAIL");
}

int main(int argc, char **argv) try {
    if (argc != 2) {
      printf("Usage: %s <repeat>\n", argv[0]);
      return 1;
    }
    const int repeat = atoi(argv[1]);
    const int batch  = kBatch;
    const int dim    = kDim;
    const int dstate = kDstate;
    const int seqlen = kSeqLen;
    const bool use_z = true; // test with SiLU gate enabled

    printf("  batch=%d  dim=%d  dstate=%d  seqlen=%d  z=%s\n\n",
           batch, dim, dstate, seqlen, use_z ? "yes" : "no");

    int N_udz  = batch * dim * seqlen;
    int N_A    = dim * dstate;
    int N_BC   = batch * dstate * seqlen;
    int N_D    = dim;
    int N_st   = batch * dim * dstate;
    int N_y    = N_udz;

    size_t SZ  = sizeof(float);

    float *h_u      = (float*)malloc(N_udz * SZ);
    float *h_delta  = (float*)malloc(N_udz * SZ);
    float *h_A      = (float*)malloc(N_A   * SZ);
    float *h_B      = (float*)malloc(N_BC  * SZ);
    float *h_C      = (float*)malloc(N_BC  * SZ);
    float *h_D      = (float*)malloc(N_D   * SZ);
    float *h_dbias  = (float*)malloc(N_D   * SZ);
    float *h_z      = (float*)malloc(N_udz * SZ);
    float *h_states0= (float*)malloc(N_st  * SZ);   // initial state (zeros)
    float *h_s_ref  = (float*)malloc(N_st  * SZ);
    float *h_s_n    = (float*)malloc(N_st  * SZ);   // base kernel states
    float *h_s_v    = (float*)malloc(N_st  * SZ);   // vllm-style  kernel states
    float *h_y_ref  = (float*)malloc(N_y   * SZ);
    float *h_y_n    = (float*)malloc(N_y   * SZ);
    float *h_y_v    = (float*)malloc(N_y   * SZ);

    srand(19937);
    fill_rand(h_u,     N_udz, -1.f,  1.f);
    fill_rand(h_delta, N_udz, -1.f,  1.f);
    fill_rand(h_A,     N_A,   -1.f,  0.f);   // A <= 0 for stability
    fill_rand(h_B,     N_BC,  -1.f,  1.f);
    fill_rand(h_C,     N_BC,  -1.f,  1.f);
    fill_rand(h_D,     N_D,   -1.f,  1.f);
    fill_rand(h_dbias, N_D,   -1.f,  1.f);
    fill_rand(h_z,     N_udz, -2.f,  2.f);
    memset(h_states0, 0, N_st * SZ);

    // reference
    memcpy(h_s_ref, h_states0, N_st * SZ);
    selective_scan_ref(h_u, h_delta, h_A, h_B, h_C, h_D, h_dbias,
                       use_z ? h_z : nullptr, /*delta_softplus=*/true,
                       batch, dim, dstate, seqlen,
                       h_s_ref, h_y_ref);

#ifdef USE_GPU
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
    sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

    float *d_u, *d_delta, *d_A, *d_B, *d_C, *d_D, *d_dbias, *d_z;
    float *d_states, *d_y;

    d_u = (float *)sycl::malloc_device(N_udz * SZ, q);
    d_delta = (float *)sycl::malloc_device(N_udz * SZ, q);
    d_A = (float *)sycl::malloc_device(N_A * SZ, q);
    d_B = (float *)sycl::malloc_device(N_BC * SZ, q);
    d_C = (float *)sycl::malloc_device(N_BC * SZ, q);
    d_D = (float *)sycl::malloc_device(N_D * SZ, q);
    d_dbias = (float *)sycl::malloc_device(N_D * SZ, q);
    d_z = (float *)sycl::malloc_device(N_udz * SZ, q);
    d_states = (float *)sycl::malloc_device(N_st * SZ, q);
    d_y = (float *)sycl::malloc_device(N_y * SZ, q);

    q.memcpy(d_u, h_u, N_udz * SZ);
    q.memcpy(d_delta, h_delta, N_udz * SZ);
    q.memcpy(d_A, h_A, N_A * SZ);
    q.memcpy(d_B, h_B, N_BC * SZ);
    q.memcpy(d_C, h_C, N_BC * SZ);
    q.memcpy(d_D, h_D, N_D * SZ);
    
    q.memcpy(d_dbias, h_dbias, N_D * SZ);
    q.memcpy(d_z, h_z, N_udz * SZ);
    
    q.memcpy(d_states, h_states0, N_st * SZ);

    int tx = (dim < 1024) ? dim : 1024;
    sycl::range<3> gws (1, (dim + tx - 1) / tx, batch * tx);
    sycl::range<3> lws (1, 1, tx);
    q.parallel_for(sycl::nd_range<3>(gws, lws), [=](sycl::nd_item<3> item) {
      selective_scan_base(d_u, d_delta, d_A, d_B, d_C, d_D,
                          d_dbias, use_z ? d_z : nullptr,
                          true, batch, dim, dstate, seqlen,
                          d_states, d_y, item);
    });

    q.memcpy(h_y_n, d_y, N_y * SZ);
    q.memcpy(h_s_n, d_states, N_st * SZ);

    // reset states
    q.memcpy(d_states, h_states0, N_st * SZ);
    const int CHUNK  = kNT * kNI;
    int n_chunks = (seqlen + CHUNK - 1) / CHUNK;
    bool even_len = (seqlen % CHUNK == 0);

    int N_carry = batch * dim * dstate * n_chunks;
    float *d_carry_a, *d_carry_b;
    d_carry_a = (float *)sycl::malloc_device(N_carry * SZ, q);
    q.memset(d_carry_a, 0, N_carry * SZ);
    d_carry_b = (float *)sycl::malloc_device(N_carry * SZ, q);
    q.memset(d_carry_b, 0, N_carry * SZ);

    SSMParamsBase p;
    p.batch   = batch;  p.dim   = dim;
    p.seqlen  = seqlen; p.dstate = dstate;
    p.n_chunks = n_chunks;
    p.u_ptr   = d_u;     p.delta_ptr      = d_delta;
    p.A_ptr   = d_A;     p.B_ptr          = d_B;
    p.C_ptr   = d_C;     p.D_ptr          = d_D;
    p.delta_bias_ptr = d_dbias;
    p.z_ptr   = use_z ? d_z : nullptr;
    p.ssm_states_ptr = d_states;
    p.out_ptr = d_y;
    // Strides (elements)
    p.u_batch_stride     = dim * seqlen;
    p.u_d_stride         = seqlen;
    p.B_batch_stride     = dstate * seqlen;
    p.B_dstate_stride    = seqlen;
    p.states_batch_stride= dim * dstate;
    p.states_d_stride    = dstate;

    launch_kernel(p, use_z, /*softplus=*/true, even_len, d_carry_a, d_carry_b, q);

    q.memcpy(h_y_v, d_y, N_y * SZ);
    q.memcpy(h_s_v, d_states, N_st * SZ);
    q.wait();
    sycl::free(d_carry_a, q);
    sycl::free(d_carry_b, q);

    const float TOL = 1e-3f;
    printf("  Results vs reference (tolerance=%.0e):\n", TOL);
    print_result("Base kernel",
                 max_abs_err(h_y_ref, h_y_n, N_y),
                 max_abs_err(h_s_ref, h_s_n, N_st), TOL);
    print_result("VLLM-style kernel",
                 max_abs_err(h_y_ref, h_y_v, N_y),
                 max_abs_err(h_s_ref, h_s_v, N_st), TOL);

    // benchmarking
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < repeat; i++) {
      q.parallel_for(sycl::nd_range<3>(gws, lws), [=](sycl::nd_item<3> item) {
        selective_scan_base(d_u, d_delta, d_A, d_B, d_C, d_D,
                            d_dbias, use_z ? d_z : nullptr,
                            true, batch, dim, dstate, seqlen,
                            d_states, d_y, item);
      });
    }
    q.wait();
    auto end = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of base kernel %f (ms)\n", (time * 1e-6f) / repeat);

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < repeat; i++) {
      int N_carry = batch * dim * dstate * n_chunks;
      float *d_carry_a, *d_carry_b;
      d_carry_a = (float *)sycl::malloc_device(N_carry * SZ, q);
      q.memset(d_carry_a, 0, N_carry * SZ);
      d_carry_b = (float *)sycl::malloc_device(N_carry * SZ, q);
      q.memset(d_carry_b, 0, N_carry * SZ);
      launch_kernel(p, use_z, /*softplus=*/true, even_len, d_carry_a, d_carry_b, q);
      q.wait();
      sycl::free(d_carry_a, q);
      sycl::free(d_carry_b, q);
    }
    end = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of vllm-style kernel %f (ms)\n", (time * 1e-6f) / repeat);

    free(h_u); free(h_delta); free(h_A); free(h_B); free(h_C);
    free(h_D); free(h_dbias); free(h_z); free(h_states0);
    free(h_s_ref); free(h_s_n); free(h_s_v);
    free(h_y_ref); free(h_y_n); free(h_y_v);

    sycl::free(d_u, q);
    sycl::free(d_delta, q);
    sycl::free(d_A, q);
    sycl::free(d_B, q);
    sycl::free(d_C, q);
    sycl::free(d_D, q);
    sycl::free(d_dbias, q);
    sycl::free(d_z, q);
    sycl::free(d_states, q);
    sycl::free(d_y, q);
    return 0;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
