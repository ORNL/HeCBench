#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <omp.h>
#include "host_utils.h"
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
    printf("Error: NUM_V_HEADS (%d) must be a multiple of NUM_HEADS (%d)\n", HV, H);
    return 1;
  }
  const int G = HV / H;
  const float scale = 1.f / sqrtf((float)K);

  printf("KDA fused recurrent forward\n");
  printf("B=%d T=%d H=%d HV=%d K=%d V=%d (G=%d), repeat=%d\n",
         B, T, H, HV, K, V, G, repeat);
  printf("dtype: q/k/v/beta/o=fp16  g/state=fp32\n");

  const size_t qk_elems   = (size_t)B * T * H  * K;
  const size_t v_elems    = (size_t)B * T * HV * V;
  const size_t g_elems    = (size_t)B * T * HV * K;
  const size_t beta_elems = (size_t)B * T * HV;
  const size_t st_elems   = (size_t)B * HV * K * V;

  float *q      = (float*) checked_malloc(qk_elems   * sizeof(float), "q");
  float *k      = (float*) checked_malloc(qk_elems   * sizeof(float), "k");
  float *v      = (float*) checked_malloc(v_elems    * sizeof(float), "v");
  float *g      = (float*) checked_malloc(g_elems    * sizeof(float), "g");
  float *beta   = (float*) checked_malloc(beta_elems * sizeof(float), "beta");
  float *o      = (float*) checked_malloc(v_elems    * sizeof(float), "output");
  float *o_ref  = (float*) checked_malloc(v_elems    * sizeof(float), "reference output");
  float *ht     = (float*) checked_malloc(st_elems   * sizeof(float), "final state");
  float *ht_ref = (float*) checked_malloc(st_elems   * sizeof(float), "reference final state");

  std::mt19937 gen(19937);
  std::uniform_real_distribution<float> qkv(-1.f, 1.f);
  std::uniform_real_distribution<float> gate(-0.25f, 0.f);
  std::uniform_real_distribution<float> bdist(0.f, 1.f);

  for (size_t i = 0; i < qk_elems; i++)   { q[i] = qkv(gen); k[i] = qkv(gen); }
  for (size_t i = 0; i < v_elems; i++)      v[i] = qkv(gen);
  for (size_t i = 0; i < g_elems; i++)      g[i] = gate(gen);
  for (size_t i = 0; i < beta_elems; i++)   beta[i] = bdist(gen);

  for (size_t base = 0; base < qk_elems; base += K) {
    float nrm = 0.f;
    for (int i = 0; i < K; i++) nrm += k[base + i] * k[base + i];
    nrm = 1.f / sqrtf(nrm + 1e-6f);
    for (int i = 0; i < K; i++) k[base + i] *= nrm;
  }

  kda_f16 *q_f16    = (kda_f16*) checked_malloc(qk_elems   * sizeof(kda_f16), "q_f16");
  kda_f16 *k_f16    = (kda_f16*) checked_malloc(qk_elems   * sizeof(kda_f16), "k_f16");
  kda_f16 *v_f16    = (kda_f16*) checked_malloc(v_elems    * sizeof(kda_f16), "v_f16");
  kda_f16 *beta_f16 = (kda_f16*) checked_malloc(beta_elems * sizeof(kda_f16), "beta_f16");
  kda_f16 *o_f16    = (kda_f16*) checked_malloc(v_elems    * sizeof(kda_f16), "o_f16");
  pack_f16(q, q_f16, qk_elems);
  pack_f16(k, k_f16, qk_elems);
  pack_f16(v, v_f16, v_elems);
  pack_f16(beta, beta_f16, beta_elems);
  unpack_f16(q_f16, q, qk_elems);
  unpack_f16(k_f16, k, qk_elems);
  unpack_f16(v_f16, v, v_elems);
  unpack_f16(beta_f16, beta, beta_elems);

  #pragma omp target enter data map(to: q_f16[0:qk_elems], k_f16[0:qk_elems], \
                                       v_f16[0:v_elems], g[0:g_elems], \
                                       beta_f16[0:beta_elems]) \
                                map(alloc: o_f16[0:v_elems], ht[0:st_elems])

  const int T_check = T <= KDA_VERIFY_T ? T : KDA_VERIFY_T;
  printf("Verify T=%d (timed T=%d)\n", T_check, T);

  fused_recurrent_kda<HEAD_K>(q_f16, k_f16, v_f16, g, beta_f16, o_f16, ht, 1,
                              scale, B, T_check, H, HV, V, G, T);

  #pragma omp target update from(o_f16[0:v_elems], ht[0:st_elems])
  unpack_f16(o_f16, o, v_elems);

  auto ref_start = std::chrono::steady_clock::now();
  reference_kda<float>(q, k, v, g, beta, o_ref, ht_ref, (double)scale,
                       B, T_check, H, HV, K, V, G, T);
  auto ref_end = std::chrono::steady_clock::now();
  double ref_ms = std::chrono::duration<double, std::milli>(ref_end - ref_start).count();
  printf("Reference (CPU) time: %.3f ms\n", ref_ms);

  const double max_err_o =
      max_abs_error_time_prefix(o, o_ref, B, T, T_check, (size_t)HV * V);
  const double max_err_h = max_abs_error(ht, ht_ref, st_elems);
  printf("Max abs error: output=%e  final_state=%e\n", max_err_o, max_err_h);

  const double tol = 1e-2;
  bool ok = (max_err_o < tol) && (max_err_h < tol);
  printf("%s\n", ok ? "PASS" : "FAIL");

  // warmup
  for (int r = 0; r < KDA_WARMUP; r++) {
    fused_recurrent_kda<HEAD_K>(q_f16, k_f16, v_f16, g, beta_f16, o_f16, ht, 0,
                                scale, B, T, H, HV, V, G);
  }

  auto start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++) {
    fused_recurrent_kda<HEAD_K>(q_f16, k_f16, v_f16, g, beta_f16, o_f16, ht, 0,
                                scale, B, T, H, HV, V, G);
  }
  auto end = std::chrono::steady_clock::now();
  double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
  printf("Average kernel execution time: %.3f ms\n", total_ms / repeat);

  #pragma omp target exit data map(delete: q_f16[0:qk_elems], k_f16[0:qk_elems], \
                                           v_f16[0:v_elems], g[0:g_elems], \
                                           beta_f16[0:beta_elems], \
                                           o_f16[0:v_elems], ht[0:st_elems])

  free(q); free(k); free(v); free(g); free(beta);
  free(q_f16); free(k_f16); free(v_f16); free(beta_f16); free(o_f16);
  free(o); free(o_ref); free(ht); free(ht_ref);

  return ok ? 0 : 1;
}
