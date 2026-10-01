// oneDNN (DNNL) 3-step MLA decode backend:
//     S = scale * Q . K^T      (GEMM1 via dnnl::matmul, scale = eltwise post-op)
//     P = softmax_row(S)       (dedicated SYCL kernel; see softmax.h)
//     O = P . V                (GEMM2 via dnnl::matmul, bf16 output)
//
// oneDNN matmul is a batched GEMM: batch dimensions are the leading tensor dims,
// and arbitrary strides let us view the KV cache transposed (K^T) or sliced
// (V = first D_V cols) with no data movement. oneDNN also has a softmax
// primitive, but it does not emit the log-sum-exp the harness verifies, so we
// keep the shared SYCL softmax kernel (which also writes LSE) for both backends.
// The QK scale is fused into GEMM1 via an eltwise-linear post-op.
//
// Requires softmax.h (bf16_t, mla_softmax) and the DIM/D_V/PAGE macros.

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include "softmax.h"

using dnnl::memory;

static memory make_usm(const memory::desc &md, const dnnl::engine &eng, void *p) {
  return dnnl::sycl_interop::make_memory(
      md, eng, dnnl::sycl_interop::memory_kind::usm, p);
}

static void launch_dnnl(sycl::queue &q, int B, int H, int max_blocks,
                        float scale, const bf16_t *dq, const bf16_t *dkv,
                        bf16_t *dout, float *dlse) {
  const int seqlen = max_blocks * PAGE;
  const size_t rows = (size_t)B * H;

  static float  *dS  = nullptr;
  static bf16_t *dP  = nullptr;
  static size_t  capS = 0;
  const size_t needS = rows * seqlen;
  if (needS > capS) {
    if (dS) sycl::free(dS, q);
    if (dP) sycl::free(dP, q);
    dS = sycl::malloc_device<float>(needS, q);
    dP = sycl::malloc_device<bf16_t>(needS, q);
    capS = needS;
  }

  // Engine/stream bound to the harness queue (built once).
  static bool inited = false;
  static dnnl::engine eng;
  static dnnl::stream strm;
  if (!inited) {
    eng  = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
    strm = dnnl::sycl_interop::make_stream(eng, q);
    inited = true;
  }

  // Cache the two matmul primitives per (B, seqlen) shape.
  static dnnl::matmul mm1, mm2;
  static memory::desc q_md, kt_md, s_md, p_md, v_md, o_md;
  static int cB = -1, cS = -1;
  if (cB != B || cS != seqlen) {
    const auto bf16 = memory::data_type::bf16;
    const auto f32  = memory::data_type::f32;

    // GEMM1: S[B,H,seqlen] = scale * Q[B,H,DIM] . (K[B,seqlen,DIM])^T
    q_md  = memory::desc({B, H, DIM},      bf16, {H * DIM, DIM, 1});
    kt_md = memory::desc({B, DIM, seqlen}, bf16, {seqlen * DIM, 1, DIM});
    s_md  = memory::desc({B, H, seqlen},   f32,  {H * seqlen, seqlen, 1});

    dnnl::post_ops po1;
    po1.append_eltwise(dnnl::algorithm::eltwise_linear, scale, 0.f);
    dnnl::primitive_attr attr1;
    attr1.set_post_ops(po1);
    mm1 = dnnl::matmul(dnnl::matmul::primitive_desc(eng, q_md, kt_md, s_md, attr1));

    // GEMM2: O[B,H,D_V] = P[B,H,seqlen] . V[B,seqlen,D_V]
    p_md = memory::desc({B, H, seqlen}, bf16, {H * seqlen, seqlen, 1});
    v_md = memory::desc({B, seqlen, D_V}, bf16, {seqlen * DIM, DIM, 1});
    o_md = memory::desc({B, H, D_V}, bf16, {H * D_V, D_V, 1});
    mm2 = dnnl::matmul(dnnl::matmul::primitive_desc(eng, p_md, v_md, o_md));

    cB = B; cS = seqlen;
  }

  mm1.execute(strm, {
      {DNNL_ARG_SRC,     make_usm(q_md,  eng, (void *)dq)},
      {DNNL_ARG_WEIGHTS, make_usm(kt_md, eng, (void *)dkv)},
      {DNNL_ARG_DST,     make_usm(s_md,  eng, dS)}});

  mla_softmax(q, dS, dP, dlse, seqlen, rows);

  mm2.execute(strm, {
      {DNNL_ARG_SRC,     make_usm(p_md, eng, dP)},
      {DNNL_ARG_WEIGHTS, make_usm(v_md, eng, (void *)dkv)},
      {DNNL_ARG_DST,     make_usm(o_md, eng, dout)}});
}
