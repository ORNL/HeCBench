#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <sycl/sycl.hpp>
#include "benchmark_utils.h"
#include "kernels.h"
#include "reference.h"

#ifndef NUM_HEADS
#define NUM_HEADS 96          // number of key/query heads (H)
#endif
#ifndef NUM_V_HEADS
#define NUM_V_HEADS 96        // number of value heads (HV), must be a multiple of H
#endif
#ifndef HEAD_K
#define HEAD_K 128            // key/query head dimension (K)
#endif
#ifndef HEAD_V
#define HEAD_V 128            // value head dimension (V)
#endif
// The thread split, V-tile, and token tile are derived from the dimensions
// above and the device, so no per-GPU tuning flags are needed.

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <batch size> <sequence length> <repeat>\n", argv[0]);
    return 1;
  }

  int B, T, repeat;
  if (!parse_positive_int(argv[1], "batch size", &B) ||
      !parse_positive_int(argv[2], "sequence length", &T) ||
      !parse_positive_int(argv[3], "repeat", &repeat))
    return 1;

  const int H  = NUM_HEADS;
  const int HV = NUM_V_HEADS;
  const int K  = HEAD_K;
  const int V  = HEAD_V;

  if (HV % H != 0) {
    printf("Error: NUM_V_HEADS (%d) must be a multiple of NUM_HEADS (%d)\n",
           HV, H);
    return 1;
  }
  const int G = HV / H;
  const float scale = 1.f / sqrtf((float)K);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  printf("KDA fused recurrent forward\n");
  printf("B=%d T=%d H=%d HV=%d K=%d V=%d (G=%d), repeat=%d\n",
         B, T, H, HV, K, V, G, repeat);
  printf("dtype: q/k/v/beta/o=bf16  g/state=fp32\n");

  const size_t qk_elems   = (size_t)B * T * H  * K;
  const size_t v_elems    = (size_t)B * T * HV * V;
  const size_t g_elems    = (size_t)B * T * HV * K;
  const size_t beta_elems = (size_t)B * T * HV;
  const size_t st_elems   = (size_t)B * HV * K * V;

  float *h_q    = (float*) checked_malloc(qk_elems   * sizeof(float), "q");
  float *h_k    = (float*) checked_malloc(qk_elems   * sizeof(float), "k");
  float *h_v    = (float*) checked_malloc(v_elems    * sizeof(float), "v");
  float *h_g    = (float*) checked_malloc(g_elems    * sizeof(float), "g");
  float *h_beta = (float*) checked_malloc(beta_elems * sizeof(float), "beta");
  float *h_o    = (float*) checked_malloc(v_elems    * sizeof(float), "output");
  float *o_ref  = (float*) checked_malloc(v_elems    * sizeof(float), "reference output");
  float *ht     = (float*) checked_malloc(st_elems   * sizeof(float), "final state");
  float *ht_ref = (float*) checked_malloc(st_elems   * sizeof(float), "reference final state");

  std::mt19937 gen(19937);
  std::uniform_real_distribution<float> qkv(-1.f, 1.f);
  std::uniform_real_distribution<float> gate(-0.25f, 0.f);
  std::uniform_real_distribution<float> bdist(0.f, 1.f);

  for (size_t i = 0; i < qk_elems; i++)   { h_q[i] = qkv(gen); h_k[i] = qkv(gen); }
  for (size_t i = 0; i < v_elems; i++)      h_v[i] = qkv(gen);
  for (size_t i = 0; i < g_elems; i++)      h_g[i] = gate(gen);
  for (size_t i = 0; i < beta_elems; i++)   h_beta[i] = bdist(gen);

  for (size_t base = 0; base < qk_elems; base += K) {
    float nrm = 0.f;
    for (int i = 0; i < K; i++) nrm += h_k[base + i] * h_k[base + i];
    nrm = 1.f / sqrtf(nrm + 1e-6f);
    for (int i = 0; i < K; i++) h_k[base + i] *= nrm;
  }

  kda_bf16 *q_bf    = (kda_bf16*) checked_malloc(qk_elems   * sizeof(kda_bf16), "q_bf16");
  kda_bf16 *k_bf    = (kda_bf16*) checked_malloc(qk_elems   * sizeof(kda_bf16), "k_bf16");
  kda_bf16 *v_bf    = (kda_bf16*) checked_malloc(v_elems    * sizeof(kda_bf16), "v_bf16");
  kda_bf16 *beta_bf = (kda_bf16*) checked_malloc(beta_elems * sizeof(kda_bf16), "beta_bf16");
  kda_bf16 *o_bf    = (kda_bf16*) checked_malloc(v_elems    * sizeof(kda_bf16), "o_bf16");
  pack_bf16(h_q, q_bf, qk_elems);
  pack_bf16(h_k, k_bf, qk_elems);
  pack_bf16(h_v, v_bf, v_elems);
  pack_bf16(h_beta, beta_bf, beta_elems);
  unpack_bf16(q_bf, h_q, qk_elems);
  unpack_bf16(k_bf, h_k, qk_elems);
  unpack_bf16(v_bf, h_v, v_elems);
  unpack_bf16(beta_bf, h_beta, beta_elems);

  kda_bf16 *d_q = sycl::malloc_device<kda_bf16>(qk_elems, q);
  kda_bf16 *d_k = sycl::malloc_device<kda_bf16>(qk_elems, q);
  kda_bf16 *d_v = sycl::malloc_device<kda_bf16>(v_elems, q);
  float *d_g = sycl::malloc_device<float>(g_elems, q);
  kda_bf16 *d_beta = sycl::malloc_device<kda_bf16>(beta_elems, q);
  kda_bf16 *d_o = sycl::malloc_device<kda_bf16>(v_elems, q);
  float *d_ht = sycl::malloc_device<float>(st_elems, q);
  if (!d_q || !d_k || !d_v || !d_g || !d_beta || !d_o || !d_ht) {
    fprintf(stderr, "Error: device allocation failed\n");
    return 1;
  }

  q.memcpy(d_q, q_bf, qk_elems * sizeof(kda_bf16));
  q.memcpy(d_k, k_bf, qk_elems * sizeof(kda_bf16));
  q.memcpy(d_v, v_bf, v_elems * sizeof(kda_bf16));
  q.memcpy(d_g, h_g, g_elems * sizeof(float));
  q.memcpy(d_beta, beta_bf, beta_elems * sizeof(kda_bf16));

  const int cus =
      q.get_device().get_info<sycl::info::device::max_compute_units>();
  const int wave = kda_device_wave(q.get_device());
  const kda_launch_plan plan = kda_plan<HEAD_K, HEAD_V>(B, HV, cus, wave);
  if (!kda::wave_ok(plan.warp) || !kda::plan_ok(plan)) {
    fprintf(stderr,
            "Error: cannot form a wave-aligned KDA launch for K=%d V=%d wave=%d\n",
            K, V, wave);
    return 1;
  }
  printf("Launch shape for %s (sub-group=%d, %d compute units): "
         "%d threads per state column, %d value columns and %d tokens per "
         "block, grid=%d block=%d\n",
         q.get_device().get_info<sycl::info::device::name>().c_str(),
         plan.warp, cus,
         plan.parts, plan.vtile, plan.unroll, plan.grid, plan.block);

  const int T_check = T <= KDA_VERIFY_T ? T : KDA_VERIFY_T;
  printf("Verify T=%d (timed T=%d)\n", T_check, T);

  kda_launch<HEAD_K, HEAD_V>(q, d_q, d_k, d_v, d_g, d_beta, d_o, d_ht, scale,
                             B, T_check, H, HV, V, G, plan.vtile, plan.warp, T);
  q.memcpy(o_bf, d_o, v_elems * sizeof(kda_bf16));
  q.memcpy(ht, d_ht, st_elems * sizeof(float));
  q.wait();
  unpack_bf16(o_bf, h_o, v_elems);

  auto ref_start = std::chrono::steady_clock::now();
  reference_kda<float>(h_q, h_k, h_v, h_g, h_beta, o_ref, ht_ref, (double)scale,
                       B, T_check, H, HV, K, V, G, T);
  auto ref_end = std::chrono::steady_clock::now();
  double ref_ms = std::chrono::duration<double, std::milli>(ref_end - ref_start).count();
  printf("Reference (CPU) time: %.3f ms\n", ref_ms);

  const double max_err_o =
      max_abs_error_time_prefix(h_o, o_ref, B, T, T_check, (size_t)HV * V);
  const double max_err_h = max_abs_error(ht, ht_ref, st_elems);
  printf("Max abs error: output=%e  final_state=%e\n", max_err_o, max_err_h);

  const double tol = 1e-2;
  bool ok = (max_err_o < tol) && (max_err_h < tol);
  printf("%s\n", ok ? "PASS" : "FAIL");

  // warmup
  for (int r = 0; r < KDA_WARMUP; r++) {
    kda_launch<HEAD_K, HEAD_V>(q, d_q, d_k, d_v, d_g, d_beta, d_o, nullptr, scale,
                               B, T, H, HV, V, G, plan.vtile, plan.warp);
  }
  q.wait();

  auto start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++) {
    kda_launch<HEAD_K, HEAD_V>(q, d_q, d_k, d_v, d_g, d_beta, d_o, nullptr, scale,
                               B, T, H, HV, V, G, plan.vtile, plan.warp);
  }
  q.wait();
  auto stop = std::chrono::steady_clock::now();
  double total_ms = std::chrono::duration<double, std::milli>(stop - start).count();
  printf("Average kernel execution time: %.3f ms\n", total_ms / repeat);

  free(h_q); free(h_k); free(h_v); free(h_g); free(h_beta);
  free(q_bf); free(k_bf); free(v_bf); free(beta_bf); free(o_bf);
  free(h_o); free(o_ref); free(ht); free(ht_ref);
  sycl::free(d_q, q); sycl::free(d_k, q); sycl::free(d_v, q);
  sycl::free(d_g, q); sycl::free(d_beta, q); sycl::free(d_o, q);
  sycl::free(d_ht, q);

  return ok ? 0 : 1;
}
