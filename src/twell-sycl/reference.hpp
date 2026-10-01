// Portable SYCL fp32 reference implementation of the gated FFN, used by
// main.cpp to validate the oneDNN bf16 path and to report gate sparsity.
//
// Written in plain SYCL with no vendor-specific features and no matrix engines,
// so it runs on any SYCL device and serves as an independent correctness
// baseline. It computes the dense gated FFN in fp32, mirroring the bf16-input,
// fp32-accumulate, bf16-gate-value semantics of the tested path:
//
//     gate = relu(x @ GATE^T);  up = x @ UP^T;  out = (bf16(gate) * up) @ DOWN
//
// The three GEMMs use a straightforward 16x16 shared-memory tiling. This is the
// validation/reporting baseline only; the high-performance path is oneDNN.
#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>

namespace ref {

using bf16 = sycl::ext::oneapi::bfloat16;

static constexpr int TS = 16;  // reference GEMM tile size

// rounds n up to a multiple of TS.
static inline size_t roundup(size_t n) { return ((n + TS - 1) / TS) * TS; }

// C[M,N] = A[M,K] @ B[N,K]  (i.e. A @ B^T), inputs bf16 upcast to fp32.
// Used for gate = A@GATE^T and up = A@UP^T (both operands K-contiguous).
template <typename T>
static sycl::event gemm_nt(sycl::queue& q, const T* A, const T* B, float* C,
                           int M, int N, int K) {
    sycl::range<2> global(roundup(M), roundup(N));
    sycl::range<2> local(TS, TS);
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 2> As({TS, TS}, h);
        sycl::local_accessor<float, 2> Bs({TS, TS}, h);
        h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) {
            const int row = it.get_global_id(0);
            const int col = it.get_global_id(1);
            const int ly = it.get_local_id(0);
            const int lx = it.get_local_id(1);
            float acc = 0.0f;
            for (int t = 0; t < K; t += TS) {
                As[ly][lx] = (row < M && t + lx < K)
                                 ? static_cast<float>(A[static_cast<long>(row) * K + t + lx])
                                 : 0.0f;
                Bs[ly][lx] = (col < N && t + ly < K)
                                 ? static_cast<float>(B[static_cast<long>(col) * K + t + ly])
                                 : 0.0f;
                it.barrier(sycl::access::fence_space::local_space);
                #pragma unroll
                for (int k = 0; k < TS; ++k) acc += As[ly][k] * Bs[k][lx];
                it.barrier(sycl::access::fence_space::local_space);
            }
            if (row < M && col < N) C[static_cast<long>(row) * N + col] = acc;
        });
    });
}

// OUT[M,hidden] = H[M,K] @ DOWN[K,hidden] ; H fp32, DOWN bf16, OUT bf16.
template <typename T>
static sycl::event gemm_nn_down(sycl::queue& q, const float* H, const T* DOWN,
                                T* OUT, int M, int hidden, int K) {
    sycl::range<2> global(roundup(M), roundup(hidden));
    sycl::range<2> local(TS, TS);
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 2> As({TS, TS}, h);
        sycl::local_accessor<float, 2> Bs({TS, TS}, h);
        h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) {
            const int row = it.get_global_id(0);
            const int col = it.get_global_id(1);
            const int ly = it.get_local_id(0);
            const int lx = it.get_local_id(1);
            float acc = 0.0f;
            for (int t = 0; t < K; t += TS) {
                As[ly][lx] = (row < M && t + lx < K)
                                 ? H[static_cast<long>(row) * K + t + lx]
                                 : 0.0f;
                Bs[ly][lx] = (col < hidden && t + ly < K)
                                 ? static_cast<float>(
                                       DOWN[static_cast<long>(t + ly) * hidden + col])
                                 : 0.0f;
                it.barrier(sycl::access::fence_space::local_space);
                #pragma unroll
                for (int k = 0; k < TS; ++k) acc += As[ly][k] * Bs[k][lx];
                it.barrier(sycl::access::fence_space::local_space);
            }
            if (row < M && col < hidden)
                OUT[static_cast<long>(row) * hidden + col] = static_cast<T>(acc);
        });
    });
}

// H[i] = float(bf16(relu(gate[i]))) * up[i] ; active[i] = gate[i] > 0.
static sycl::event fuse_gate_up(sycl::queue& q, const float* gate_raw,
                                const float* up_raw, float* H, uint8_t* active,
                                long total) {
    return q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> i) {
        const float gp = gate_raw[i];
        const bool pos = gp > 0.0f;
        const float gate = pos ? static_cast<float>(static_cast<bf16>(gp)) : 0.0f;
        H[i] = gate * up_raw[i];
        active[i] = pos ? 1u : 0u;
    });
}

// launches the portable reference gated FFN: gate + up GEMMs, fuse, down GEMM.
template <typename T>
static void launch_reference(sycl::queue& q, const T* A, const T* GATE,
                             const T* UP, const T* DOWN, float* gate_raw,
                             float* up_raw, float* H, uint8_t* active, T* OUT,
                             int M, int hidden, int intermediate) {
    gemm_nt<T>(q, A, GATE, gate_raw, M, intermediate, hidden);
    gemm_nt<T>(q, A, UP, up_raw, M, intermediate, hidden);
    const long total = static_cast<long>(M) * intermediate;
    fuse_gate_up(q, gate_raw, up_raw, H, active, total);
    gemm_nn_down<T>(q, H, DOWN, OUT, M, hidden, intermediate);
    q.wait();
}

// per (row, 256-tile) count of positive gate activations (gate-density report).
static void tile_count(sycl::queue& q, const uint8_t* active, int* counts, int M,
                       int intermediate) {
    const int tiles = intermediate / 256;
    const long total = static_cast<long>(M) * tiles;
    q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> t) {
         const int m = static_cast<int>(t / tiles);
         const int ti = static_cast<int>(t % tiles);
         const uint8_t* row = active + static_cast<long>(m) * intermediate + ti * 256;
         int c = 0;
         for (int i = 0; i < 256; ++i) c += row[i];
         counts[t] = c;
     }).wait();
}

}  // namespace ref
