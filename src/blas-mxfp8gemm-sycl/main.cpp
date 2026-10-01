#include <chrono>
#include <unordered_map>
#include "helper.h"

using namespace dnnl;

// Block-scaled MXFP8 matmul with the oneDNN matmul primitive.
//
//   D(M,N) = (A(M,K) * a_scale) @ (B(N,K) * b_scale)^T
// A and B are FP8 (f8_e4m3), one byte per element along K.
// The block scales are E8M0 (e8m0) with one scale per 32 elements along the K
// dimension, expressed through per-argument scale attributes with group size 32
// (matching the OCP microscaling format). Output D is BF16.
//
// This follows the oneDNN mxfp8 matmul example
// (https://uxlfoundation.github.io/oneDNN/example_mxfp_matmul.cpp.html) but
// keeps the BF16 output and the benchmark structure of blas-mxfp8gemm-cuda.
//
// A is stored (M,K) row-major and B is stored (N,K) row-major. B is described
// as {K,N} with strides {1,K} so that it is interpreted as B^T, producing
// D = A @ B^T of shape (M,N).
//
// Returns false if oneDNN has no mxfp8 matmul implementation for this platform.
bool OnednnMxfp8Matmul(const int repeat, Mxfp8TestBench &tb) {
    const int m = tb.m, n = tb.n, k = tb.k;

    auto a_md = memory::desc({m, k}, memory::data_type::f8_e4m3, memory::dims{k, 1});
    auto b_md = memory::desc({k, n}, memory::data_type::f8_e4m3, memory::dims{1, k});
    auto c_md = memory::desc({m, n}, memory::data_type::bf16,    memory::dims{n, 1});

    // Per-argument E8M0 block scales along K (group size = SCALE_BLOCK_SIZE).
    // mask selects both dimensions so scales vary per row and per K-block.
    const int mask = (1 << 1) | (1 << 0);
    primitive_attr attr;
    attr.set_scales(DNNL_ARG_SRC, mask, {1, Mxfp8TestBench::SCALE_BLOCK_SIZE},
                    memory::data_type::e8m0);
    attr.set_scales(DNNL_ARG_WEIGHTS, mask, {Mxfp8TestBench::SCALE_BLOCK_SIZE, 1},
                    memory::data_type::e8m0);

    matmul::primitive_desc pd;
    try {
        pd = matmul::primitive_desc(tb.engine, a_md, b_md, c_md, attr);
    } catch (dnnl::error &e) {
        if (e.status == dnnl_unimplemented) {
            printf("no mxfp8 matmul implementation available for current configuration\n");
            return false;
        }
        throw;
    }
    auto prim = matmul(pd);

    auto a_mem = sycl_interop::make_memory(a_md, tb.engine, sycl_interop::memory_kind::usm, tb.Adev);
    auto b_mem = sycl_interop::make_memory(b_md, tb.engine, sycl_interop::memory_kind::usm, tb.Bdev);
    auto c_mem = sycl_interop::make_memory(c_md, tb.engine, sycl_interop::memory_kind::usm, tb.Ddev);

    auto a_scale_md = memory::desc({(memory::dim)tb.aScaleBytes}, memory::data_type::e8m0, memory::dims{1});
    auto b_scale_md = memory::desc({(memory::dim)tb.bScaleBytes}, memory::data_type::e8m0, memory::dims{1});
    auto a_scale_mem = sycl_interop::make_memory(a_scale_md, tb.engine, sycl_interop::memory_kind::usm, tb.AscaleDev);
    auto b_scale_mem = sycl_interop::make_memory(b_scale_md, tb.engine, sycl_interop::memory_kind::usm, tb.BscaleDev);

    std::unordered_map<int, memory> args = {
        {DNNL_ARG_SRC, a_mem},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, a_scale_mem},
        {DNNL_ARG_WEIGHTS, b_mem},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, b_scale_mem},
        {DNNL_ARG_DST, c_mem},
    };

    // Warmup.
    prim.execute(tb.stream, args);
    tb.stream.wait();

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
        prim.execute(tb.stream, args);
    }
    tb.stream.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    auto ns = (time / repeat);
    printf("Average oneDNN matmul execution time %10.3f (us) | ", ns * 1e-3f);
    printf("Average oneDNN matmul performance %.1f (TFLOPS)\n", 2.f * m * k * n / ns * 1e-3f);

    return true;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <repeat>\n", argv[0]);
        return 1;
    }
    const int repeat = atoi(argv[1]);

#ifdef USE_GPU
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
    sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif
    printf("Device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    const int shapes[6][3] = {{16384, 8192, 1280},
                              {16384, 1024, 8192},
                              {16384, 8192, 7168},
                              {16384, 3584, 8192},
                              {8192, 8192, 8192},
                              {16384, 16384, 16384}};

    for (int i = 0; i < 6; i++) {
        int m = shapes[i][0], n = shapes[i][1], k = shapes[i][2];
        printf("Matrix dimension (M, N, K) = (%d, %d, %d)\n", m, n, k);

        Mxfp8TestBench props(q, m, n, k, 1.0f, 0.0f);

        bool executed = OnednnMxfp8Matmul(repeat, props);
        if (!executed) break;

        props.verify();
    }

    return 0;
}
