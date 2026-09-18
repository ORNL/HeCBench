#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <cuda.h>
#include "benchmark_utils.h"
#include "kernels.h"
#include "reference.h"

// Override via EXTRA_CFLAGS, e.g. Kimi-K3: make EXTRA_CFLAGS="-DNUM_HEADS=96 -DNUM_V_HEADS=96"
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
// above and the device

#define CHECK(call)                                                          \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(err));                                      \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

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

  printf("KDA fused recurrent forward\n");
  printf("B=%d T=%d H=%d HV=%d K=%d V=%d (G=%d), repeat=%d\n",
         B, T, H, HV, K, V, G, repeat);
  printf("dtype: q/k/v/beta/o=bf16  g/state=fp32\n");

  const size_t qk_elems   = (size_t)B * T * H  * K;   // q, k
  const size_t v_elems    = (size_t)B * T * HV * V;   // v, o
  const size_t g_elems    = (size_t)B * T * HV * K;   // g
  const size_t beta_elems = (size_t)B * T * HV;       // beta
  const size_t st_elems   = (size_t)B * HV * K * V;   // final state

  float *q      = (float*) checked_malloc(qk_elems   * sizeof(float), "q");
  float *k      = (float*) checked_malloc(qk_elems   * sizeof(float), "k");
  float *v      = (float*) checked_malloc(v_elems    * sizeof(float), "v");
  float *g      = (float*) checked_malloc(g_elems    * sizeof(float), "g");
  float *beta   = (float*) checked_malloc(beta_elems * sizeof(float), "beta");
  float *o      = (float*) checked_malloc(v_elems    * sizeof(float), "output");
  float *o_ref  = (float*) checked_malloc(v_elems    * sizeof(float), "reference output");
  float *ht     = (float*) checked_malloc(st_elems   * sizeof(float), "final state");
  float *ht_ref = (float*) checked_malloc(st_elems   * sizeof(float), "reference final state");

  // Initialize inputs with moderate magnitudes so the recurrence stays bounded.
  std::mt19937 gen(19937);
  std::uniform_real_distribution<float> qkv(-1.f, 1.f);   // q, k, v
  std::uniform_real_distribution<float> gate(-0.25f, 0.f); // log-decay (exp in (0.78,1))
  std::uniform_real_distribution<float> bdist(0.f, 1.f);   // beta

  for (size_t i = 0; i < qk_elems; i++)   { q[i] = qkv(gen); k[i] = qkv(gen); }
  for (size_t i = 0; i < v_elems; i++)      v[i] = qkv(gen);
  for (size_t i = 0; i < g_elems; i++)      g[i] = gate(gen);
  for (size_t i = 0; i < beta_elems; i++)   beta[i] = bdist(gen);

  // L2-normalize each k head-vector so the delta-rule state update stays a
  // contraction (standard in DeltaNet/KDA); otherwise the recurrence diverges.
  for (size_t base = 0; base < qk_elems; base += K) {
    float nrm = 0.f;
    for (int i = 0; i < K; i++) nrm += k[base + i] * k[base + i];
    nrm = 1.f / sqrtf(nrm + 1e-6f);
    for (int i = 0; i < K; i++) k[base + i] *= nrm;
  }

  nv_bfloat16 *q_bf   = (nv_bfloat16*) checked_malloc(qk_elems   * sizeof(nv_bfloat16), "q_bf16");
  nv_bfloat16 *k_bf   = (nv_bfloat16*) checked_malloc(qk_elems   * sizeof(nv_bfloat16), "k_bf16");
  nv_bfloat16 *v_bf   = (nv_bfloat16*) checked_malloc(v_elems    * sizeof(nv_bfloat16), "v_bf16");
  nv_bfloat16 *beta_bf= (nv_bfloat16*) checked_malloc(beta_elems * sizeof(nv_bfloat16), "beta_bf16");
  nv_bfloat16 *o_bf   = (nv_bfloat16*) checked_malloc(v_elems    * sizeof(nv_bfloat16), "o_bf16");
  pack_bf16(q, q_bf, qk_elems);
  pack_bf16(k, k_bf, qk_elems);
  pack_bf16(v, v_bf, v_elems);
  pack_bf16(beta, beta_bf, beta_elems);
  unpack_bf16(q_bf, q, qk_elems);
  unpack_bf16(k_bf, k, qk_elems);
  unpack_bf16(v_bf, v, v_elems);
  unpack_bf16(beta_bf, beta, beta_elems);

  nv_bfloat16 *d_q, *d_k, *d_v, *d_beta, *d_o;
  float *d_g, *d_ht;
  CHECK(cudaMalloc((void**)&d_q,    qk_elems   * sizeof(nv_bfloat16)));
  CHECK(cudaMalloc((void**)&d_k,    qk_elems   * sizeof(nv_bfloat16)));
  CHECK(cudaMalloc((void**)&d_v,    v_elems    * sizeof(nv_bfloat16)));
  CHECK(cudaMalloc((void**)&d_g,    g_elems    * sizeof(float)));
  CHECK(cudaMalloc((void**)&d_beta, beta_elems * sizeof(nv_bfloat16)));
  CHECK(cudaMalloc((void**)&d_o,    v_elems    * sizeof(nv_bfloat16)));
  CHECK(cudaMalloc((void**)&d_ht,   st_elems   * sizeof(float)));

  CHECK(cudaMemcpy(d_q,    q_bf,    qk_elems   * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_k,    k_bf,    qk_elems   * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_v,    v_bf,    v_elems    * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_g,    g,       g_elems    * sizeof(float),       cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_beta, beta_bf, beta_elems * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));

  int device = 0;
  cudaDeviceProp prop;
  CHECK(cudaGetDevice(&device));
  CHECK(cudaGetDeviceProperties(&prop, device));
  const kda_launch_plan plan =
      kda_plan<HEAD_K, HEAD_V>(B, HV, prop.multiProcessorCount, prop.warpSize);
  if (!kda::wave_ok(plan.warp) || !kda::plan_ok(plan)) {
    fprintf(stderr,
            "Error: cannot form a wave-aligned KDA launch for K=%d V=%d wave=%d\n",
            K, V, plan.warp);
    return 1;
  }
  printf("Launch shape for %s (sm_%d%d, %d SMs): "
         "%d threads per state column, %d value columns and %d tokens per "
         "block, wave=%d, grid=%d block=%d\n",
         prop.name, prop.major, prop.minor, prop.multiProcessorCount,
         plan.parts, plan.vtile, plan.unroll, plan.warp, plan.grid, plan.block);

  const int T_check = T <= KDA_VERIFY_T ? T : KDA_VERIFY_T;
  printf("Verify T=%d (timed T=%d)\n", T_check, T);

  kda_launch<HEAD_K, HEAD_V>(d_q, d_k, d_v, d_g, d_beta, d_o, d_ht, scale,
                             B, T_check, H, HV, V, G, plan.vtile, T);
  CHECK(cudaGetLastError());
  CHECK(cudaMemcpy(o_bf, d_o,  v_elems  * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(ht,   d_ht, st_elems * sizeof(float),       cudaMemcpyDeviceToHost));
  unpack_bf16(o_bf, o, v_elems);

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
    kda_launch<HEAD_K, HEAD_V>(d_q, d_k, d_v, d_g, d_beta, d_o, nullptr, scale,
                               B, T, H, HV, V, G, plan.vtile);
  }
  CHECK(cudaDeviceSynchronize());

  auto start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++) {
    kda_launch<HEAD_K, HEAD_V>(d_q, d_k, d_v, d_g, d_beta, d_o, nullptr, scale,
                               B, T, H, HV, V, G, plan.vtile);
  }
  CHECK(cudaDeviceSynchronize());
  auto stop = std::chrono::steady_clock::now();
  double total_ms = std::chrono::duration<double, std::milli>(stop - start).count();
  printf("Average kernel execution time: %.3f ms\n", total_ms / repeat);

  free(q); free(k); free(v); free(g); free(beta);
  free(q_bf); free(k_bf); free(v_bf); free(beta_bf); free(o_bf);
  free(o); free(o_ref); free(ht); free(ht_ref);
  CHECK(cudaFree(d_q)); CHECK(cudaFree(d_k)); CHECK(cudaFree(d_v)); CHECK(cudaFree(d_g));
  CHECK(cudaFree(d_beta)); CHECK(cudaFree(d_o)); CHECK(cudaFree(d_ht));

  return ok ? 0 : 1;
}
