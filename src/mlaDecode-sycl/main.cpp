/*
 * MLA decode, MQA mode (DeepSeek dense shapes):
 * HEAD_DIM_K = 576, HEAD_DIM_V = 512
 * Paged KV cache, PAGE_BLOCK_SIZE = 64. BF16 in/out, FP32 LSE.
 *
 * SYCL port. The two batched GEMMs run through a library backend -- oneMKL
 * (default) and/or oneDNN -- while the row softmax + LSE is a dedicated SYCL
 * kernel (neither library fuses softmax into the GEMM). Select backends with
 * -DUSE_ONEMKL and/or -DUSE_ONEDNN; if neither is set, both are built.
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <random>
#include <algorithm>
#include <sycl/sycl.hpp>

#include "reference.h"
#include "softmax.h"

#if !defined(USE_ONEMKL) && !defined(USE_ONEDNN)
#define USE_ONEMKL
#define USE_ONEDNN
#endif

#ifdef USE_ONEMKL
#include "kernel.h"
#endif
#ifdef USE_ONEDNN
#include "kernel_dnn.h"
#endif

static inline float b2f_host(bf16_t v) { return (float)v; }

static void run(sycl::queue &q, int B, int H, int seqlen, int repeat, bool verify) {
  const float scale = 1.0f / std::sqrt((float)DIM);
  const int page_per_seq = (seqlen + PAGE - 1) / PAGE;
  const int max_blocks = page_per_seq;
  const int num_pages = B * page_per_seq;
  printf("B=%d H=%d seqlen=%d num_pages=%d (DIM=%d, D_V=%d, PAGE=%d)\n",
         B, H, seqlen, num_pages, DIM, D_V, PAGE);

  size_t qElems   = (size_t)B * H * DIM;
  size_t kvElems  = (size_t)num_pages * PAGE * DIM;
  size_t oElems   = (size_t)B * H * D_V;
  size_t lseElems = (size_t)B * H;

  std::mt19937 rng(19937);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);

  std::vector<bf16_t> hq(qElems), hkv(kvElems);
  std::vector<float> hqf(qElems), hkvf(kvElems);
  for (size_t i = 0; i < qElems; i++)  {
    bf16_t v = bf16_t(dist(rng)); hq[i] = v; hqf[i] = b2f_host(v);
  }
  for (size_t i = 0; i < kvElems; i++) {
    bf16_t v = bf16_t(dist(rng)); hkv[i] = v; hkvf[i] = b2f_host(v);
  }

  bf16_t *dq   = sycl::malloc_device<bf16_t>(qElems, q);
  bf16_t *dkv  = sycl::malloc_device<bf16_t>(kvElems, q);
  bf16_t *dout = sycl::malloc_device<bf16_t>(oElems, q);
  float  *dlse = sycl::malloc_device<float>(lseElems, q);
  q.memcpy(dq, hq.data(), qElems * sizeof(bf16_t));
  q.memcpy(dkv, hkv.data(), kvElems * sizeof(bf16_t));
  q.wait();

  // Attention FLOPs: QK (2*DIM) + PV (2*D_V) per (request, head, key).
  double flops = 2.0 * B * H * (double)seqlen * (DIM + D_V);
  // Effective KV bytes: head-batched kernels read the KV cache once per request.
  double bytes = (double)B * seqlen * DIM * sizeof(bf16_t);

  std::vector<float> ref, refLse;
  if (verify) {
    ref.resize(oElems); refLse.resize(lseElems);
    reference(hqf, hkvf, ref, refLse, B, H, seqlen, scale);
  }

  auto verify_out = [&](const char *name) {
    if (!verify) return;
    std::vector<bf16_t> hout(oElems);
    std::vector<float> hlse(lseElems);
    q.memcpy(hout.data(), dout, oElems * sizeof(bf16_t));
    q.memcpy(hlse.data(), dlse, lseElems * sizeof(float));
    q.wait();
    double maxAbs = 0.0, maxRel = 0.0, maxLse = 0.0;
    for (size_t i = 0; i < oElems; i++) {
      double err = std::fabs(b2f_host(hout[i]) - ref[i]);
      maxAbs = std::max(maxAbs, err);
      maxRel = std::max(maxRel, err / (std::fabs(ref[i]) + 5e-2));
    }
    for (size_t i = 0; i < lseElems; i++)
      maxLse = std::max(maxLse, (double)std::fabs(hlse[i] - refLse[i]));
    bool ok = maxAbs < 5e-3 && maxLse < 5e-2;
    printf("  [%s] Verify: out max abs err %.4g, max rel err %.4g, lse max abs err %.4g -> %s\n",
           name, maxAbs, maxRel, maxLse, ok ? "PASS" : "FAIL");
  };

  auto bench = [&](const char *name, auto launch_fn) {
    q.memset(dout, 0, oElems * sizeof(bf16_t)).wait();
    launch_fn();
    q.wait();
    verify_out(name);

    for (int w = 0; w < 100; w++) launch_fn();   // warmup
    q.wait();

    auto start = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; r++) launch_fn();
    q.wait();
    auto end = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count() / repeat;
    printf("  [%s] Average kernel time %.3f ms | %.1f GFLOP/s | %.1f GB/s\n",
           name, ms, flops / (ms * 1e6), bytes / (ms * 1e6));
  };

  // benchmark backend(s)
#ifdef USE_ONEMKL
  bench("oneMKL", [&]{ launch(q, B, H, max_blocks, scale, dq, dkv, dout, dlse); });
#endif
#ifdef USE_ONEDNN
  bench("oneDNN", [&]{ launch_dnnl(q, B, H, max_blocks, scale, dq, dkv, dout, dlse); });
#endif

  sycl::free(dq, q); sycl::free(dkv, q);
  sycl::free(dout, q); sycl::free(dlse, q);
}

int main(int argc, char **argv) {
  int repeat = (argc > 1) ? atoi(argv[1]) : 100;
  const int H = 128;   // number of query heads

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif
  printf("Running on %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

  run(q, 64,  H, 1024, repeat, true);
  run(q, 128, H, 4096, repeat, true);
  run(q, 32,  H, 8192, repeat, true);
  return 0;
}
