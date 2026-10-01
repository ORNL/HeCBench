/*
 * MLA decode, MQA mode (DeepSeek dense shapes):
 * HEAD_DIM_K = 576, HEAD_DIM_V = 512
 * Paged KV cache, PAGE_BLOCK_SIZE = 64. BF16 in/out, FP32 LSE.
 *
 * HIP port (hipBLAS + hipBLASLt) of the CUDA reference implementation.
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <random>
#include <algorithm>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

// The reference defines DIM, D_V, PAGE, NEG_LARGE
#include "reference.h"
#include "kernel.h"
#include "kernel_lt.h"

static inline float b2f_host(bf16_t v) { return __bfloat162float(v); }

static void run(int B, int H, int seqlen, int repeat, bool verify) {
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
    bf16_t v = __float2bfloat16(dist(rng)); hq[i] = v; hqf[i] = b2f_host(v);
  }
  for (size_t i = 0; i < kvElems; i++) {
    bf16_t v = __float2bfloat16(dist(rng)); hkv[i] = v; hkvf[i] = b2f_host(v);
  }

  bf16_t *dq, *dkv, *dout; float *dlse;
  CHECK(hipMalloc(&dq, qElems * sizeof(bf16_t)));
  CHECK(hipMalloc(&dkv, kvElems * sizeof(bf16_t)));
  CHECK(hipMalloc(&dout, oElems * sizeof(bf16_t)));
  CHECK(hipMalloc(&dlse, lseElems * sizeof(float)));
  CHECK(hipMemcpy(dq, hq.data(), qElems * sizeof(bf16_t), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(dkv, hkv.data(), kvElems * sizeof(bf16_t), hipMemcpyHostToDevice));

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
    CHECK(hipMemcpy(hout.data(), dout, oElems * sizeof(bf16_t), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(hlse.data(), dlse, lseElems * sizeof(float), hipMemcpyDeviceToHost));
    // BF16 outputs carry ~2^-8 (~0.4%) relative error, and MLA outputs are
    // small weighted averages, so many elements are near zero: judge by
    // absolute error against the full [-1,1] value range, plus LSE abs error.
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
    CHECK(hipMemset(dout, 0, oElems * sizeof(bf16_t)));
    launch_fn();
    CHECK(hipDeviceSynchronize());
    verify_out(name);

    for (int w = 0; w < 100; w++) launch_fn();   // warmup
    CHECK(hipDeviceSynchronize());

    auto start = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; r++) launch_fn();
    CHECK(hipDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count() / repeat;
    printf("  [%s] Average kernel time %.3f ms | %.1f GFLOP/s | %.1f GB/s\n",
           name, ms, flops / (ms * 1e6), bytes / (ms * 1e6));
  };

  // benchmark kernel(s)
  bench("hipblas",   [&]{ launch(B, H, max_blocks, scale, dq, dkv, dout, dlse); });
  bench("hipblasLt", [&]{ launch_lt(B, H, max_blocks, scale, dq, dkv, dout, dlse); });

  CHECK(hipFree(dq)); CHECK(hipFree(dkv));
  CHECK(hipFree(dout)); CHECK(hipFree(dlse));
}

int main(int argc, char **argv) {
  int repeat = (argc > 1) ? atoi(argv[1]) : 100;
  const int H = 128;   // number of query heads

  run(64,  H, 1024, repeat, true);
  run(128, H, 4096, repeat, true);
  run(32,  H, 8192, repeat, true);
  return 0;
}
