#ifndef FLASH_MLA_REFERENCE_H
#define FLASH_MLA_REFERENCE_H

// Plain FP32 CPU reference for the FlashMLA dense decode kernel. Kept simple
// and obviously correct (a direct transcription of the attention math) so the
// GPU kernels can be validated against it.
//
// Requires the MLA layout macros (DIM, D_V, PAGE) and NEG_LARGE to be defined
// before this header is included (see main.cu).

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstddef>

// MLA latent dimensions (DeepSeek dense decode shapes).
#define D_V     512   // value / output width (the "nope" part of the latent)
#define D_TAIL   64   // "rope" tail width
#define DIM     (D_V + D_TAIL)   // 576 = HEAD_DIM_K
#define PAGE     64   // PAGE_BLOCK_SIZE
#define NEG_LARGE (-3.4028234663852886e38f)

// For one query token t (decode => one query per request) and head h:
//   score_j = scale * (Q[h] . K[j])            for j in [0, seqlen)
//   P       = softmax_j(score_j)
//   Out[h]  = sum_j P_j * V[j]                  (V[j] = K[j][0:D_V], the "nope" part)
//   lse[h]  = max_j + log( sum_j exp(score_j - max_j) )
//
// Uses the benchmark's layout assumption (see kernel.h): identity paging and a
// uniform seqlen, so request t's KV is contiguous at kv[(t*seqlen + j) * DIM].
static void reference(const std::vector<float> &q, const std::vector<float> &kv,
                      std::vector<float> &out, std::vector<float> &lse,
                      int B, int H, int seqlen, float scale) {
#ifdef _OPENMP
  #pragma omp parallel for schedule(dynamic)
#endif
  for (int t = 0; t < B; t++) {
    std::vector<float> score(seqlen);
    for (int h = 0; h < H; h++) {
      const float *qh = &q[((size_t)t * H + h) * DIM];
      float m = NEG_LARGE;
      for (int j = 0; j < seqlen; j++) {
        const float *krow = &kv[((size_t)t * seqlen + j) * DIM];
        float s = 0.f;
        for (int d = 0; d < DIM; d++) s += qh[d] * krow[d];
        s *= scale;
        score[j] = s;
        m = std::max(m, s);
      }
      float l = 0.f;
      for (int j = 0; j < seqlen; j++) { score[j] = std::exp(score[j] - m); l += score[j]; }
      float linv = (l > 0.f) ? 1.f / l : 0.f;
      float *orow = &out[((size_t)t * H + h) * D_V];
      for (int d = 0; d < D_V; d++) {
        float a = 0.f;
        for (int j = 0; j < seqlen; j++)
          a += score[j] * kv[((size_t)t * seqlen + j) * DIM + d];
        orow[d] = a * linv;
      }
      lse[(size_t)t * H + h] = (l > 0.f) ? (m + std::log(l)) : NEG_LARGE;
    }
  }
}

#endif  // FLASH_MLA_REFERENCE_H
