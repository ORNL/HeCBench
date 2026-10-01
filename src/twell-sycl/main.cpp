// TwELL-shaped gated MLP inference, SYCL + oneDNN (Intel XMX matrix engine).
//
// Modeled per-layer operation (fused gated FFN):
//     gate = relu(x @ GATE^T)
//     up   = x @ UP^T
//     out  = (gate * up) @ DOWN
//
// The high-performance path expresses the block as three oneDNN matmul
// primitives that run on the GPU matrix engine (Intel Xe XMX / DPAS) in bf16
// with fp32 accumulation. Two fusions keep memory traffic low:
//   * the gate matmul fuses a relu eltwise post-op, and
//   * the up matmul fuses a binary-mul post-op with the gate tensor, so
//     h = up * relu(gate) is produced without a separate elementwise pass.
// Weights are pre-reordered once per layer into the matmul's preferred XMX
// layout (weights created with format_tag::any) so every timed execution hits
// the fastest matrix-engine kernel.
//
// Two modes (run back-to-back, mirroring the CUDA/HIP benchmarks):
//   validate   : oneDNN bf16 gated MLP vs an independent SYCL fp32 reference.
//   benchmark  : performance over an L-layer stack.
//
// Kernel shape constraints:
//   * hidden (OUT_DIM) must be exactly 2048
//   * intermediate (FEATURE_DIM) must be 5632 or 8192
//   * tokens M = batch * seq must be divisible by 256

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"

#include "reference.hpp"
#include "twell.hpp"

using bf16 = sycl::ext::oneapi::bfloat16;
namespace dn = dnnl;

// ---------------------------------------------------------------------------
// Deterministic counter-based hash fill (matches the CUDA/HIP benchmarks):
//     v = (u01 - 0.5) * scale + offset,  u01 in [0,1)
// ---------------------------------------------------------------------------
template <typename T>
static void fill(sycl::queue& q, T* p, size_t n, unsigned seed, float scale,
                 float offset) {
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
         size_t i = idx[0];
         unsigned h = static_cast<unsigned>(i) * 2654435761u ^ seed;
         h ^= h >> 13;
         h *= 0x5bd1e995u;
         h ^= h >> 15;
         float u01 = static_cast<float>(h & 0xFFFFu) / 65535.0f;
         p[i] = static_cast<T>((u01 - 0.5f) * scale + offset);
     }).wait();
}

template <typename T>
static T* alloc_and_fill(sycl::queue& q, size_t n, unsigned seed, float scale,
                         float offset) {
    T* ptr = sycl::malloc_device<T>(n, q);
    fill<T>(q, ptr, n, seed, scale, offset);
    return ptr;
}

// ---------------------------------------------------------------------------
// oneDNN gated MLP (XMX matrix-engine path)
// ---------------------------------------------------------------------------
template <typename T>
struct GatedMLP {
    static constexpr const char* name = "oneDNN-dense";
    dn::engine eng;
    dn::stream strm;
    int M, K, Nf, Hd;
    dn::memory::data_type dt;

    dn::matmul mm_gate, mm_up, mm_down;
    dn::memory A_mem, gate_mem, h_mem, out_mem;
    // per-layer weights, pre-reordered into the matmul's preferred XMX layout
    std::vector<dn::memory> gateW, upW, downW;

    GatedMLP(sycl::queue& q, int M_, int hidden, int intermediate)
        : M(M_), K(hidden), Nf(intermediate), Hd(hidden) {
        dt = std::is_same<T, bf16>::value ? dn::memory::data_type::bf16
                                          : dn::memory::data_type::f32;
        eng = dn::sycl_interop::make_engine(q.get_device(), q.get_context());
        strm = dn::sycl_interop::make_stream(eng, q);

        using md = dn::memory::desc;
        using tag = dn::memory::format_tag;
        //using dims = dn::memory::dims;

        // activation tensors (plain row-major)
        md md_A({M, K}, dt, tag::ab);
        md md_gate({M, Nf}, dt, tag::ab);
        md md_h({M, Nf}, dt, tag::ab);
        md md_out({M, Hd}, dt, tag::ab);

        // weights, layout chosen by oneDNN for the fastest matrix-engine kernel
        md md_gateW_any({K, Nf}, dt, tag::any);
        md md_upW_any({K, Nf}, dt, tag::any);
        md md_downW_any({Nf, Hd}, dt, tag::any);

        // gate = relu(A @ GATE^T)
        dn::post_ops po_relu;
        po_relu.append_eltwise(dn::algorithm::eltwise_relu, 0.0f, 0.0f);
        dn::primitive_attr attr_relu;
        attr_relu.set_post_ops(po_relu);
        dn::matmul::primitive_desc pd_gate(eng, md_A, md_gateW_any, md_gate,
                                           attr_relu);

        // h = (A @ UP^T) * gate   (binary-mul post-op fuses the gate multiply)
        dn::post_ops po_mul;
        po_mul.append_binary(dn::algorithm::binary_mul, md_gate);
        dn::primitive_attr attr_mul;
        attr_mul.set_post_ops(po_mul);
        dn::matmul::primitive_desc pd_up(eng, md_A, md_upW_any, md_h, attr_mul);

        // out = h @ DOWN
        dn::matmul::primitive_desc pd_down(eng, md_h, md_downW_any, md_out);

        mm_gate = dn::matmul(pd_gate);
        mm_up = dn::matmul(pd_up);
        mm_down = dn::matmul(pd_down);

        // activation memories: A/out rebased per layer; gate/h are scratch
        A_mem = dn::memory(md_A, eng, DNNL_MEMORY_NONE);
        out_mem = dn::memory(md_out, eng, DNNL_MEMORY_NONE);
        gate_mem = dn::memory(pd_gate.dst_desc(), eng);
        h_mem = dn::memory(pd_up.dst_desc(), eng);

        gate_pd_ = pd_gate;
        up_pd_ = pd_up;
        down_pd_ = pd_down;
    }

    dn::matmul::primitive_desc gate_pd_, up_pd_, down_pd_;

    // reorders one layer's raw weights (GATE/UP/DOWN, bf16 row-major (N x K) /
    // (K x hidden)) into the XMX-preferred layout and caches them.
    void add_layer(T* GATE, T* UP, T* DOWN) {
        using md = dn::memory::desc;
        using dims = dn::memory::dims;
        // GATE/UP are (Nf x K) row-major; the matmul wants B = weight^T with
        // logical shape (K x Nf), which is the same memory viewed transposed.
        md user_gateW({K, Nf}, dt, dims{1, K});
        md user_upW({K, Nf}, dt, dims{1, K});
        // DOWN is (Nf x hidden) row-major = the natural (K x N) weight layout.
        md user_downW({Nf, Hd}, dt, dims{Hd, 1});

        dn::memory ug(user_gateW, eng, GATE);
        dn::memory uu(user_upW, eng, UP);
        dn::memory ud(user_downW, eng, DOWN);

        dn::memory rg(gate_pd_.weights_desc(), eng);
        dn::memory ru(up_pd_.weights_desc(), eng);
        dn::memory rd(down_pd_.weights_desc(), eng);

        dn::reorder(ug, rg).execute(strm, ug, rg);
        dn::reorder(uu, ru).execute(strm, uu, ru);
        dn::reorder(ud, rd).execute(strm, ud, rd);

        gateW.push_back(rg);
        upW.push_back(ru);
        downW.push_back(rd);
    }

    // runs one gated MLP layer: in (M x hidden) -> out (M x hidden).
    void run(int layer, T* in, T* out) {
        A_mem.set_data_handle(in);
        out_mem.set_data_handle(out);

        mm_gate.execute(strm, {{DNNL_ARG_SRC, A_mem},
                               {DNNL_ARG_WEIGHTS, gateW[layer]},
                               {DNNL_ARG_DST, gate_mem}});
        mm_up.execute(
            strm,
            {{DNNL_ARG_SRC, A_mem},
             {DNNL_ARG_WEIGHTS, upW[layer]},
             {DNNL_ARG_DST, h_mem},
             {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, gate_mem}});
        mm_down.execute(strm, {{DNNL_ARG_SRC, h_mem},
                               {DNNL_ARG_WEIGHTS, downW[layer]},
                               {DNNL_ARG_DST, out_mem}});
    }
};

// ---------------------------------------------------------------------------
// TwELL packed-sparse gated MLP: XMX gate GEMM (oneDNN) -> pack (D2T) -> sparse
// gated T2D. Instantiated for bf16 only (the packed format is bf16).
//
// HIGH_PRECISION selects between the two TwELL kernels (bf16-product
// default vs fp32 high-precision); both keep the packed-sparse structure.
// ---------------------------------------------------------------------------
template <typename T, bool HIGH_PRECISION>
struct TwellGatedMLPImpl {
    static constexpr const char* name =
        HIGH_PRECISION ? "TwELL-sparse-hp" : "TwELL-sparse";
    sycl::queue* q_;
    dn::engine eng;
    dn::stream strm;
    int M, K, Nf;  // K = hidden = OUT_DIM
    dn::memory::data_type dt;

    dn::matmul mm_gate;
    dn::matmul::primitive_desc gate_pd_;
    dn::memory A_mem, gate_mem;   // gate_mem: dense relu(gate), (M x Nf) bf16
    bf16* gate_dense = nullptr;
    uint32_t* packed = nullptr;
    std::vector<dn::memory> gateW;      // reordered gate weights (XMX layout)
    std::vector<T*> upW, downW;         // raw up/down weights (FEATURE x hidden)

    TwellGatedMLPImpl(sycl::queue& q, int M_, int hidden, int intermediate)
        : q_(&q), M(M_), K(hidden), Nf(intermediate) {
        static_assert(std::is_same<T, bf16>::value,
                      "TwELL packed path is bf16-only");
        dt = dn::memory::data_type::bf16;
        eng = dn::sycl_interop::make_engine(q.get_device(), q.get_context());
        strm = dn::sycl_interop::make_stream(eng, q);

        using md = dn::memory::desc;
        using tag = dn::memory::format_tag;
        md md_A({M, K}, dt, tag::ab);
        md md_gate({M, Nf}, dt, tag::ab);
        md md_gateW_any({K, Nf}, dt, tag::any);

        dn::post_ops po_relu;
        po_relu.append_eltwise(dn::algorithm::eltwise_relu, 0.0f, 0.0f);
        dn::primitive_attr attr_relu;
        attr_relu.set_post_ops(po_relu);
        gate_pd_ = dn::matmul::primitive_desc(eng, md_A, md_gateW_any, md_gate,
                                              attr_relu);
        mm_gate = dn::matmul(gate_pd_);

        A_mem = dn::memory(md_A, eng, DNNL_MEMORY_NONE);
        gate_mem = dn::memory(gate_pd_.dst_desc(), eng);
        gate_dense = static_cast<bf16*>(gate_mem.get_data_handle());

        const int NUM_T_n = Nf / twell::T_n;
        packed = sycl::malloc_device<uint32_t>(
            static_cast<size_t>(M) * NUM_T_n * twell::T_n_compressed, q);
    }

    void add_layer(T* GATE, T* UP, T* DOWN) {
        using md = dn::memory::desc;
        using dims = dn::memory::dims;
        // GATE (Nf x K) row-major viewed transposed as (K x Nf) weight
        md user_gateW({K, Nf}, dt, dims{1, K});
        dn::memory ug(user_gateW, eng, GATE);
        dn::memory rg(gate_pd_.weights_desc(), eng);
        dn::reorder(ug, rg).execute(strm, ug, rg);
        gateW.push_back(rg);
        upW.push_back(UP);
        downW.push_back(DOWN);
    }

    void run(int layer, T* in, T* out) {
        // D2T: gate = relu(in @ GATE^T) on the XMX matrix engine, then pack.
        A_mem.set_data_handle(in);
        mm_gate.execute(strm, {{DNNL_ARG_SRC, A_mem},
                               {DNNL_ARG_WEIGHTS, gateW[layer]},
                               {DNNL_ARG_DST, gate_mem}});
        twell::pack_gate(*q_, gate_dense, packed, M, Nf);
        // gated T2D: sparse up/down projection over active gate features only.
        if constexpr (HIGH_PRECISION) {
            twell::gated_t2d_high_precision(
                *q_, reinterpret_cast<const bf16*>(in), packed,
                reinterpret_cast<const bf16*>(upW[layer]),
                reinterpret_cast<const bf16*>(downW[layer]),
                reinterpret_cast<bf16*>(out), M, Nf, K);
        } else {
            twell::gated_t2d(*q_, reinterpret_cast<const bf16*>(in), packed,
                             reinterpret_cast<const bf16*>(upW[layer]),
                             reinterpret_cast<const bf16*>(downW[layer]),
                             reinterpret_cast<bf16*>(out), M, Nf, K);
        }
    }
};

// single-type-parameter aliases so both variants fit the `template <typename>
// class MLP` interface used by run_validate / run_benchmark.
template <typename T>
using TwellGatedMLP = TwellGatedMLPImpl<T, false>;
template <typename T>
using TwellGatedMLPHighPrecision = TwellGatedMLPImpl<T, true>;

// ---------------------------------------------------------------------------
// Arg parsing
// ---------------------------------------------------------------------------
struct Args {
    int hidden = 2048;
    int intermediate = 8192;
    int batch = 2;
    int seq = 2048;
    int layers = 16;
    int reps = 50;
    int warmup = 10;
    bool high_precision = false;
    float gate_bias = -0.0013f;
};

static void parse_args(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", k.c_str());
                std::exit(1);
            }
            return argv[++i];
        };
        if (k == "--hidden") a.hidden = std::atoi(next());
        else if (k == "--intermediate") a.intermediate = std::atoi(next());
        else if (k == "--batch") a.batch = std::atoi(next());
        else if (k == "--seq") a.seq = std::atoi(next());
        else if (k == "--layers") a.layers = std::atoi(next());
        else if (k == "--reps") a.reps = std::atoi(next());
        else if (k == "--warmup") a.warmup = std::atoi(next());
        else if (k == "--high-precision") a.high_precision = true;
        else if (k == "--gate-bias") a.gate_bias = static_cast<float>(std::atof(next()));
        else if (k == "--help" || k == "-h") {
            printf(
                "Usage: %s [options]\n"
                "  --hidden N         hidden size / OUT_DIM (must be 2048)\n"
                "  --intermediate N   intermediate / FEATURE_DIM (5632 or 8192)\n"
                "  --batch N          batch size (default 2)\n"
                "  --seq N            sequence length (default 2048)\n"
                "  --layers N         gated MLP layers in stack (default 16)\n"
                "  --reps N           timed reps (default 50)\n"
                "  --warmup N         warmup reps (default 10)\n"
                "  --high-precision   use fp32 matmuls instead of bf16\n"
                "  --gate-bias F      negative GATE bias for sparsity (default -0.0013)\n",
                argv[0]);
            std::exit(0);
        } else {
            fprintf(stderr, "unknown argument: %s\n", k.c_str());
            std::exit(1);
        }
    }
}

static void validate_shape(const Args& a, long long M) {
    if (a.hidden != 2048) {
        fprintf(stderr, "error: --hidden must be 2048 (got %d)\n", a.hidden);
        std::exit(1);
    }
    if (a.intermediate % 256 != 0 ||
        (a.intermediate / 256 != 22 && a.intermediate / 256 != 32)) {
        fprintf(stderr, "error: --intermediate must be 5632 or 8192 (got %d)\n",
                a.intermediate);
        std::exit(1);
    }
    if (M % 256 != 0) {
        fprintf(stderr, "error: batch*seq must be divisible by 256 (got %lld)\n", M);
        std::exit(1);
    }
}

static constexpr float kWeightScale = 0.10f;
static constexpr float kActScale = 0.50f;
static constexpr float kActOffset = 0.25f;

// ---------------------------------------------------------------------------
// Validation: oneDNN bf16 gated MLP vs SYCL fp32 reference
// ---------------------------------------------------------------------------
template <typename T, template <typename> class MLP>
static int run_validate(sycl::queue& q, const Args& args) {
    const size_t M = static_cast<size_t>(args.batch) * args.seq;
    validate_shape(args, M);
    const int hidden = args.hidden, intermediate = args.intermediate;

    printf("Device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    printf("Validate: M=%zu hidden=%d intermediate=%d gate_bias=%.5f precision=%s\n",
           M, hidden, intermediate, args.gate_bias,
           std::is_same<T, bf16>::value ? "bf16" : "fp32");

    const size_t wN = static_cast<size_t>(intermediate) * hidden;
    T* GATE = alloc_and_fill<T>(q, wN, 0x11u, kWeightScale, args.gate_bias);
    T* UP = alloc_and_fill<T>(q, wN, 0x22u, kWeightScale, 0.0f);
    T* DOWN = alloc_and_fill<T>(q, wN, 0x33u, kWeightScale, 0.0f);

    const size_t actN = M * hidden;
    T* A = alloc_and_fill<T>(q, actN, 0x44u, kActScale, kActOffset);

    // Reference buffers
    float* gate_raw = sycl::malloc_device<float>(M * intermediate, q);
    float* up_raw = sycl::malloc_device<float>(M * intermediate, q);
    float* H = sycl::malloc_device<float>(M * intermediate, q);
    uint8_t* active = sycl::malloc_device<uint8_t>(M * intermediate, q);
    T* OUT_ref = sycl::malloc_device<T>(actN, q);

    // TwELL buffers
    T* OUT_dnn = sycl::malloc_device<T>(actN, q);

    // ---- reference ----
    ref::launch_reference<T>(q, A, GATE, UP, DOWN, gate_raw, up_raw, H, active,
                             OUT_ref, M, hidden, intermediate);

    const int tiles = intermediate / 256;
    int* tile_counts = sycl::malloc_device<int>(M * tiles, q);
    ref::tile_count(q, active, tile_counts, M, intermediate);
    std::vector<int> tc(M * tiles);
    q.memcpy(tc.data(), tile_counts, tc.size() * sizeof(int)).wait();
    long long sum_active = 0;
    int max_tile = 0;
    for (int v : tc) { sum_active += v; if (v > max_tile) max_tile = v; }
    double density = static_cast<double>(sum_active) / (static_cast<double>(M) * intermediate);
    printf("Gate sparsity: density=%.3f%%  avg_nnz/tile=%.1f  max_nnz/tile=%d\n",
           density * 100.0,
           static_cast<double>(sum_active) / (static_cast<double>(M) * tiles), max_tile);

    // ---- tested gated MLP ----
    MLP<T> mlp(q, M, hidden, intermediate);
    mlp.add_layer(GATE, UP, DOWN);
    mlp.run(0, A, OUT_dnn);

    // ---- compare on host (fp32) ----
    std::vector<T> ref_h(actN), dn_h(actN);
    q.memcpy(ref_h.data(), OUT_ref, actN * sizeof(T));
    q.memcpy(dn_h.data(), OUT_dnn, actN * sizeof(T));
    q.wait();

    double max_abs = 0.0, sum_abs = 0.0, sum_ref = 0.0, max_ref = 0.0;
    for (size_t i = 0; i < actN; ++i) {
        double r = static_cast<float>(ref_h[i]);
        double t = static_cast<float>(dn_h[i]);
        double d = std::fabs(r - t);
        max_abs = std::max(max_abs, d);
        sum_abs += d;
        sum_ref += std::fabs(r);
        max_ref = std::max(max_ref, std::fabs(r));
    }
    double mean_abs = sum_abs / actN;
    double mean_ref = sum_ref / actN;
    printf("\nCorrectness (%s %s vs fp32 reference)\n", MLP<T>::name,
           std::is_same<T, bf16>::value ? "bf16" : "fp32");
    printf("  reference   : mean|out|=%.5f  max|out|=%.5f\n", mean_ref, max_ref);
    printf("  abs error   : mean=%.6f  max=%.6f\n", mean_abs, max_abs);
    printf("  rel error   : mean=%.4f%%  max(vs mean|out|)=%.4f%%\n",
           (mean_abs / (mean_ref + 1e-12)) * 100.0,
           (max_abs / (mean_ref + 1e-12)) * 100.0);
    double rel_mean = mean_abs / (mean_ref + 1e-12);
    bool pass = rel_mean < 0.03;
    printf("  RESULT      : %s\n", pass ? "PASS" : "FAIL");

    sycl::free(GATE, q); sycl::free(UP, q); sycl::free(DOWN, q); sycl::free(A, q);
    sycl::free(gate_raw, q); sycl::free(up_raw, q); sycl::free(H, q);
    sycl::free(active, q); sycl::free(OUT_ref, q); sycl::free(OUT_dnn, q);
    sycl::free(tile_counts, q);
    return pass ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Performance benchmark over an L-layer stack
// ---------------------------------------------------------------------------
template <typename T, template <typename> class MLP>
static int run_benchmark(sycl::queue& q, const Args& args) {
    const long long M = static_cast<long long>(args.batch) * args.seq;
    validate_shape(args, M);
    const int hidden = args.hidden;
    const int intermediate = args.intermediate;

    printf("Device: %s\n",
           q.get_device().get_info<sycl::info::device::name>().c_str());
    printf(
        "Config: tokens(M)=%lld hidden=%d intermediate=%d layers=%d reps=%d "
        "warmup=%d precision=%s gate_bias=%.5f\n",
        M, hidden, intermediate, args.layers, args.reps, args.warmup,
        std::is_same<T, bf16>::value ? "bf16" : "fp32", args.gate_bias);

    const size_t wN = static_cast<size_t>(intermediate) * hidden;
    MLP<T> mlp(q, M, hidden, intermediate);
    std::vector<T*> gate(args.layers), up(args.layers), down(args.layers);
    for (int l = 0; l < args.layers; ++l) {
        gate[l] = alloc_and_fill<T>(q, wN, 0x1000u + l, kWeightScale, args.gate_bias);
        up[l] = alloc_and_fill<T>(q, wN, 0x2000u + l, kWeightScale, 0.0f);
        down[l] = alloc_and_fill<T>(q, wN, 0x3000u + l, kWeightScale, 0.0f);
        mlp.add_layer(gate[l], up[l], down[l]);
    }

    const size_t actN = M * hidden;
    T* act_a = alloc_and_fill<T>(q, actN, 0xABCDu, kActScale, kActOffset);
    T* act_b = sycl::malloc_device<T>(actN, q);

    auto run_stack = [&]() {
        T* in = act_a;
        T* out = act_b;
        for (int l = 0; l < args.layers; ++l) {
            mlp.run(l, in, out);
            std::swap(in, out);
        }
    };

    printf("Warmup (%d reps)...\n", args.warmup);
    for (int r = 0; r < args.warmup; ++r) run_stack();
    q.wait();

    printf("Timing (%d reps)...\n", args.reps);
    auto start = std::chrono::steady_clock::now();
    for (int r = 0; r < args.reps; ++r) run_stack();
    q.wait();
    auto end = std::chrono::steady_clock::now();

    double total_ms =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-6;
    const double per_rep_ms = total_ms / args.reps;
    const double per_layer_ms = per_rep_ms / args.layers;
    const double tokens_per_s = (M * 1000.0) / per_rep_ms;
    const double dense_flops_per_rep =
        static_cast<double>(M) * intermediate * (6.0 * hidden + 1.0) * args.layers;
    const double tflops = (dense_flops_per_rep / (per_rep_ms / 1000.0)) / 1e12;

    printf("\nResults\n");
    printf("  total time            : %.3f ms (%d reps)\n", total_ms, args.reps);
    printf("  per forward (stack)   : %.4f ms\n", per_rep_ms);
    printf("  per gated MLP layer   : %.4f ms\n", per_layer_ms);
    printf("  input throughput      : %.1f tokens/s\n", tokens_per_s);
    printf("  dense-equivalent perf : %.2f TFLOP/s\n", tflops);

    for (int l = 0; l < args.layers; ++l) {
        sycl::free(gate[l], q); sycl::free(up[l], q); sycl::free(down[l], q);
    }
    sycl::free(act_a, q); sycl::free(act_b, q);
    return 0;
}

template <typename T, template <typename> class MLP>
static int run_all(sycl::queue& q, const Args& args) {
    if (run_validate<T, MLP>(q, args) == 0) return run_benchmark<T, MLP>(q, args);
    return 1;
}

int main(int argc, char** argv) {
    Args args;
    parse_args(argc, argv, args);
    sycl::queue q{sycl::gpu_selector_v,
                  sycl::property::queue::in_order()};
    // Both modes use the TwELL packed-sparse path
    if (args.high_precision)
        return run_all<bf16, TwellGatedMLPHighPrecision>(q, args);
    return run_all<bf16, TwellGatedMLP>(q, args);
}
