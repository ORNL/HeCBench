#ifndef KDA_REFERENCE_H
#define KDA_REFERENCE_H

#include <cmath>
#include <vector>

// CPU reference for the KDA (Kimi Delta Attention) fused recurrent forward pass.
//
// This mirrors `naive_recurrent_kda` from
//   https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/naive.py
//
// KDA is a *linear-attention* variant. Instead of forming an explicit T x T
// attention matrix (as in softmax attention), it maintains a compact recurrent
// "state" matrix S of shape [K, V] that is updated token-by-token as we scan
// over the sequence. This makes the cost linear in sequence length T rather
// than quadratic, while still letting every past token influence the output.
//
// Two ideas layered on top of plain linear attention:
//   * Per-dimension decay gates `g` (a.k.a. gated linear attention): each row of
//     the state is multiplicatively decayed every step, so old information fades.
//   * A "delta rule" update using `beta`: rather than blindly adding the new
//     key/value outer product, we first subtract what the current key already
//     predicts (u = v - k.S), then write back the *error* u scaled by beta. This
//     is the classic delta-rule / fast-weight programming correction and is what
//     gives KDA (and DeltaNet-style models) better memory behavior.
//
// -------------------------------------------------------------------------
// Tensor shapes (row-major / PyTorch-contiguous; last dim is fastest-varying)
// -------------------------------------------------------------------------
//   q, k    : [B, T, H,  K]   query/key,   K = head dim of the key space
//   v, o    : [B, T, HV, V]   value/output, V = head dim of the value space
//   g       : [B, T, HV, K]   per-dimension decay gates, stored in log-space
//   beta    : [B, T, HV]      per-(token,head) scalar learning-rate for delta rule
//   S(state): [B, HV, K, V]   recurrent state, one [K,V] matrix per (batch, head)
//
//   B  = batch size
//   T  = sequence length (number of time steps we scan over)
//   H  = number of query/key heads
//   HV = number of value heads (>= H). Grouped attention: several value heads
//        share one query/key head. G = HV / H is the group size, and value head
//        hv maps to query/key head h = hv / G. (When H == HV, G == 1.)
//
// -------------------------------------------------------------------------
// Recurrence for each (b, hv), with h = hv / G, per time step i (S starts at 0)
// -------------------------------------------------------------------------
//   S  = S * exp(g_i)                                  # per-row (per-k) decay
//   u  = v_i - (k_i . S)                               # u[vd] = sum_k k_i[k]*S[k,vd]
//   S  = S + outer(beta_i * k_i, u)                    # S[k,vd] += beta*k_i[k]*u[vd]
//   o_i = (scale * q_i) . S                            # o[vd]  = sum_k q_i[k]*S[k,vd]
//
//  The four math steps (one token)
// 1. Each row of S is multiplied by exp(g[k]), a number usually a bit less than 1.
//
//  for each key channel k:
//      for each value channel vd:
//          S[k][vd] = S[k][vd] * exp(g[k])
//  Plain English: old memory fades; different key channels can fade at different speeds.
//
// 2. Predict, then form the error. Treat S as a linear memory: “given this key, what value do I already retrieve?”
//  for each vd:
//      prediction[vd] = sum over k of  ( k[k] * S[k][vd] )
//      u[vd]          = v[vd] - prediction[vd]
//  Plain English: u is “what is still missing.” If S already predicts v, u is 0 and you write nothing.
//
// 3. Write the error
//
//  for each k:
//      for each vd:
//          S[k][vd] = S[k][vd] + beta * k[k] * u[vd]
//
//  Plain English:  update: every column vd gets u[vd] written into it
//
// 4. Read with the query from the updated S
//
//  for each vd:
//      o[vd] = sum over k of  ( scale * q[k] * S[k][vd] )
//
//  Plain English: mix the rows of memory using the query; that mix is the output for this token.

// Notes:
//   * exp(g_i) turns the log-space gate into a multiplicative decay in (0, 1].
//   * `scale` is the usual attention scaling (e.g. 1/sqrt(K)) applied to q.
//   * The output o_i uses the *updated* S (after the current token is written).
//
// The computation is carried out in double precision to provide a high-accuracy
// golden result for verifying the single-precision GPU kernel.
template <typename T_in>
void reference_kda(
    const T_in* q,     // [B, T, H,  K]
    const T_in* k,     // [B, T, H,  K]
    const T_in* v,     // [B, T, HV, V]
    const T_in* g,     // [B, T, HV, K]
    const T_in* beta,  // [B, T, HV]
    T_in* o,           // [B, T, HV, V]  (output)
    T_in* h_final,     // [B, HV, K, V]  (output, may be nullptr)
    const double scale,
    const int B, const int T, const int H, const int HV,
    const int K, const int V, const int G,
    const int T_stride = -1)
{
  // T_stride is the allocated time dimension of the [B, T_stride, ...] layout.
  // T is how many tokens to scan. T_stride <= 0 means packed (stride == T).
  const int t_stride = T_stride > 0 ? T_stride : T;

  // Each (batch b, value head hv) pair owns an independent recurrence, so they
  // can all be processed in parallel. `collapse(2)` flattens the two loops into
  // one iteration space for the OpenMP scheduler.
  #pragma omp parallel for collapse(2) schedule(static)
  for (int b = 0; b < B; b++) {
    for (int hv = 0; hv < HV; hv++) {
      // Value head hv shares query/key head h (grouped attention).
      const int h = hv / G;

      // Recurrent state S[K][V] for this (b, hv), stored row-major (row = k
      // index, col = v index) and initialized to zero (empty memory).
      std::vector<double> S((size_t)K * V, 0.0);
      // Scratch for the delta-rule correction u (length V), recomputed per step.
      std::vector<double> u(V);

      // Sequential scan over time: step t depends on the state left by t-1.
      for (int t = 0; t < T; t++) {
        // Pointers to this step's slices. q/k are indexed by head h (K-dim),
        // while v/g/beta/o are indexed by value head hv (V- or K-dim). The
        // index math reproduces the row-major offsets of the shapes above.
        size_t t_in = (size_t)b * t_stride + t;
        const T_in* q_i = q + (t_in * H + h) * K;   // [K]
        const T_in* k_i = k + (t_in * H + h) * K;   // [K]
        const T_in* g_i = g + (t_in * HV + hv) * K; // [K] log-decay
        const T_in* v_i = v + (t_in * HV + hv) * V; // [V]
        const double b_i = (double)beta[t_in * HV + hv]; // scalar
        T_in* o_i = o + (t_in * HV + hv) * V;       // [V] output

        // Steps 1 & 2, fused: decay every state row by exp(g) and simultaneously
        // build u = v_i - k_i . S using the freshly decayed S.
        //   u starts as a copy of v_i...
        for (int vd = 0; vd < V; vd++) u[vd] = (double)v_i[vd];
        //   ...then for each key dimension kk we decay row kk of S and subtract
        //   its contribution k_i[kk] * S[kk, :] from u (the delta-rule "prediction").
        for (int kk = 0; kk < K; kk++) {
          const double decay = std::exp((double)g_i[kk]); // log-space -> (0,1]
          const double kv = (double)k_i[kk];
          double* Srow = S.data() + (size_t)kk * V;        // row kk = S[kk, :]
          for (int vd = 0; vd < V; vd++) {
            Srow[vd] *= decay;         // per-dimension gated decay of the state
            u[vd] -= kv * Srow[vd];    // accumulate -(k_i . S) into u
          }
        }

        // Steps 3 & 4, fused: write the delta-rule update into S and read the
        // output o_i = (scale*q_i) . S from the *updated* state.
        for (int vd = 0; vd < V; vd++) o_i[vd] = (T_in)0;    // clear accumulator
        for (int kk = 0; kk < K; kk++) {
          const double bk = b_i * (double)k_i[kk];   // beta-scaled key component
          const double qk = scale * (double)q_i[kk]; // scaled query component
          double* Srow = S.data() + (size_t)kk * V;
          for (int vd = 0; vd < V; vd++) {
            Srow[vd] += bk * u[vd];    // rank-1 delta update: outer(beta*k, u)
            // Accumulate this row's contribution to the output (dot with q).
            o_i[vd] = (T_in)((double)o_i[vd] + qk * Srow[vd]);
          }
        }
      }

      // Optionally export the final recurrent state (e.g. for chunked / carried
      // inference). Layout matches S[B, HV, K, V].
      if (h_final != nullptr) {
        T_in* Hf = h_final + ((size_t)b * HV + hv) * K * V;
        for (size_t idx = 0; idx < (size_t)K * V; idx++)
          Hf[idx] = (T_in)S[idx];
      }
    }
  }
}

#endif  // KDA_REFERENCE_H
