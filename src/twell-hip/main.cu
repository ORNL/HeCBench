// TwELL sparse MLP inference.
//
// Modeled per-layer operation (fused gated FFN):
//     gate = relu(x @ GATE^T)          (D2T -> packed sparse hidden)
//     up   = x @ UP^T
//     out  = (gate * up) @ DOWN        (fused gated packed-to-dense)
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

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include "twell_modules/torch_compat.h"  // maps at::BFloat16 -> __hip_bfloat16 (no libtorch)
#include "reference.h"                   // reference gated FFN (validation baseline)

// ---------------------------------------------------------------------------
// TwELL kernel entry points (defined in twell_modules/*.cu)
// ---------------------------------------------------------------------------
namespace TWELL_D2T {
void create_d2t_layer_128x256x64TS8(
    const int layer_number, at::BFloat16* B_d, const int K, const int N);
void ensure_d2t_layer_shape_128x256x64TS8(const int layer_number, const size_t M);
void destroy_all_d2t_layers();
}  // namespace TWELL_D2T

namespace TWELL_MLP {
void run_gated_mlp_layer_128x256x64TS8(
    const int layer_number, at::BFloat16* A_d, at::BFloat16* up_weight_d,
    at::BFloat16* down_weight_d, uint32_t* C_packed_d, at::BFloat16* out_d,
    const size_t M, const int FEATURE_DIM, const int OUT_DIM);

void run_gated_mlp_layer_128x256x64TS8_high_precision(
    const int layer_number, at::BFloat16* A_d, at::BFloat16* up_weight_d,
    at::BFloat16* down_weight_d, uint32_t* C_packed_d, at::BFloat16* out_d,
    const size_t M, const int FEATURE_DIM, const int OUT_DIM);
}  // namespace TWELL_MLP

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        hipError_t _err = (expr);                                              \
        if (_err != hipSuccess) {                                              \
            fprintf(stderr, "HIP error %s at %s:%d: %s\n", #expr, __FILE__,    \
                    __LINE__, hipGetErrorString(_err));                        \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// Deterministic counter-based hash fill: v = (u01 - 0.5) * scale + offset,
// mapped to bf16. Avoids cuRAND and large host buffers.
__global__ void fill_bf16_kernel(__nv_bfloat16* p, size_t n, unsigned seed,
                                 float scale, float offset) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned h = static_cast<unsigned>(i) * 2654435761u ^ seed;
    h ^= h >> 13;
    h *= 0x5bd1e995u;
    h ^= h >> 15;
    float u01 = static_cast<float>(h & 0xFFFFu) / 65535.0f;
    p[i] = __float2bfloat16((u01 - 0.5f) * scale + offset);
}

static at::BFloat16* alloc_and_fill(size_t num_elems, unsigned seed,
                                    float scale, float offset) {
    at::BFloat16* ptr = nullptr;
    CUDA_CHECK(hipMalloc(&ptr, num_elems * sizeof(at::BFloat16)));
    const int threads = 256;
    const size_t blocks = (num_elems + threads - 1) / threads;
    fill_bf16_kernel<<<static_cast<unsigned>(blocks), threads>>>(
        reinterpret_cast<__nv_bfloat16*>(ptr), num_elems, seed, scale, offset);
    CUDA_CHECK(hipGetLastError());
    return ptr;
}

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
    float gate_bias = -0.0013f;  // negative bias -> sparse gate activations
};

static int parse_int(const char* v) { return std::atoi(v); }
static float parse_float(const char* v) { return static_cast<float>(std::atof(v)); }

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
        if (k == "--hidden") a.hidden = parse_int(next());
        else if (k == "--intermediate") a.intermediate = parse_int(next());
        else if (k == "--batch") a.batch = parse_int(next());
        else if (k == "--seq") a.seq = parse_int(next());
        else if (k == "--layers") a.layers = parse_int(next());
        else if (k == "--reps") a.reps = parse_int(next());
        else if (k == "--warmup") a.warmup = parse_int(next());
        else if (k == "--high-precision") a.high_precision = true;
        else if (k == "--gate-bias") a.gate_bias = parse_float(next());
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
                "  --high-precision   use fp32 fused gated path\n"
                "  --gate-bias F      negative GATE bias for sparsity (default -0.0013)\n",
                argv[0]);
            std::exit(0);
        } else {
            fprintf(stderr, "unknown argument: %s\n", k.c_str());
            std::exit(1);
        }
    }
}

static void validate_shape(const Args& a, size_t M) {
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
        fprintf(stderr, "error: batch*seq must be divisible by 256 (got %zu)\n",
                M);
        std::exit(1);
    }
}

// weight amplitude (uniform half-width) for GATE/UP/DOWN and activations
static constexpr float kWeightScale = 0.10f;   // -> U[-0.05, 0.05]
static constexpr float kActScale = 0.50f;      // activations in [0, 0.5]
static constexpr float kActOffset = 0.25f;

static int run_validate(const Args& args) {
    const size_t M = static_cast<size_t>(args.batch) * args.seq;
    validate_shape(args, M);
    const int hidden = args.hidden;
    const int intermediate = args.intermediate;

    hipDeviceProp_t prop{};
    CUDA_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("Device: %s (%s, %d CUs)\n", prop.name, prop.gcnArchName,
           prop.multiProcessorCount);
    printf(
        "Validate: M=%zu hidden=%d intermediate=%d gate_bias=%.5f precision=%s\n",
        M, hidden, intermediate, args.gate_bias,
        args.high_precision ? "fp32" : "bf16");

    const size_t wN = static_cast<size_t>(intermediate) * hidden;
    at::BFloat16* GATE = alloc_and_fill(wN, 0x11u, kWeightScale, args.gate_bias);
    at::BFloat16* UP = alloc_and_fill(wN, 0x22u, kWeightScale, 0.0f);
    at::BFloat16* DOWN = alloc_and_fill(wN, 0x33u, kWeightScale, 0.0f);

    const size_t actN = M * hidden;
    at::BFloat16* A = alloc_and_fill(actN, 0x44u, kActScale, kActOffset);
    // TwELL overwrites its input tensor map address per call; keep a pristine
    // copy of A for the reference so both see identical inputs.
    at::BFloat16* A_twell = nullptr;
    CUDA_CHECK(hipMalloc(&A_twell, actN * sizeof(at::BFloat16)));
    CUDA_CHECK(hipMemcpy(A_twell, A, actN * sizeof(at::BFloat16),
                          hipMemcpyDeviceToDevice));

    // Reference buffers
    float* H = nullptr;
    CUDA_CHECK(hipMalloc(&H, M * intermediate * sizeof(float)));
    float* gate_raw = nullptr;
    CUDA_CHECK(hipMalloc(&gate_raw, M * intermediate * sizeof(float)));
    float* up_raw = nullptr;
    CUDA_CHECK(hipMalloc(&up_raw, M * intermediate * sizeof(float)));
    uint8_t* active = nullptr;
    CUDA_CHECK(hipMalloc(&active, M * intermediate * sizeof(uint8_t)));
    at::BFloat16* OUT_ref = nullptr;
    CUDA_CHECK(hipMalloc(&OUT_ref, actN * sizeof(at::BFloat16)));

    // TwELL buffers
    at::BFloat16* OUT_twell = nullptr;
    CUDA_CHECK(hipMalloc(&OUT_twell, actN * sizeof(at::BFloat16)));
    uint32_t* packed = nullptr;
    CUDA_CHECK(hipMalloc(&packed, M * (intermediate / 8) * sizeof(uint32_t)));

    // ---- Reference ----
    const int T = 256;
    auto grid = [&](long n) { return static_cast<unsigned>((n + T - 1) / T); };
    launch_reference(reinterpret_cast<__nv_bfloat16*>(A),
                     reinterpret_cast<__nv_bfloat16*>(GATE),
                     reinterpret_cast<__nv_bfloat16*>(UP),
                     reinterpret_cast<__nv_bfloat16*>(DOWN), gate_raw, up_raw, H,
                     active, reinterpret_cast<__nv_bfloat16*>(OUT_ref), M,
                     hidden, intermediate);

    // ---- Sparsity / overflow check ----
    const int tiles = intermediate / 256;
    int* tile_counts = nullptr;
    CUDA_CHECK(hipMalloc(&tile_counts, M * tiles * sizeof(int)));
    tile_count_kernel<<<grid(static_cast<long>(M) * tiles), T>>>(active, tile_counts, M, intermediate);
    CUDA_CHECK(hipGetLastError());
    std::vector<int> tc(M * tiles);
    CUDA_CHECK(hipMemcpy(tc.data(), tile_counts, tc.size() * sizeof(int), hipMemcpyDeviceToHost));
    long long sum_active = 0;
    int max_tile = 0;
    for (int v : tc) { sum_active += v; if (v > max_tile) max_tile = v; }
    double density = static_cast<double>(sum_active) / (static_cast<double>(M) * intermediate);
    const int payload_slots = 256 / 8 - 1;  // 31
    printf("Gate sparsity: density=%.3f%%  avg_nnz/tile=%.1f  max_nnz/tile=%d (budget=%d)\n",
           density * 100.0, static_cast<double>(sum_active) / (static_cast<double>(M) * tiles),
           max_tile, payload_slots);
    bool overflow = max_tile > payload_slots;
    if (overflow) {
        printf("WARNING: max_nnz/tile exceeds the %d-slot packed budget; TwELL is\n"
               "         lossy in this regime. Increase |--gate-bias| for a valid test.\n",
               payload_slots);
    }

    // ---- TwELL ----
    TWELL_D2T::create_d2t_layer_128x256x64TS8(0, GATE, hidden, intermediate);
    TWELL_D2T::ensure_d2t_layer_shape_128x256x64TS8(0, M);
    if (args.high_precision) {
        TWELL_MLP::run_gated_mlp_layer_128x256x64TS8_high_precision(
            0, A_twell, UP, DOWN, packed, OUT_twell, M, intermediate, hidden);
    } else {
        TWELL_MLP::run_gated_mlp_layer_128x256x64TS8(
            0, A_twell, UP, DOWN, packed, OUT_twell, M, intermediate, hidden);
    }
    CUDA_CHECK(hipDeviceSynchronize());

    // ---- Compare on host (fp32) ----
    std::vector<uint16_t> ref_h(actN), tw_h(actN);
    CUDA_CHECK(hipMemcpy(ref_h.data(), OUT_ref, actN * sizeof(uint16_t), hipMemcpyDeviceToHost));
    CUDA_CHECK(hipMemcpy(tw_h.data(), OUT_twell, actN * sizeof(uint16_t), hipMemcpyDeviceToHost));
    auto bf16_to_f32 = [](uint16_t b) {
        uint32_t bits = static_cast<uint32_t>(b) << 16;
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
    };
    double max_abs = 0.0, sum_abs = 0.0, sum_ref = 0.0, max_ref = 0.0;
    double max_rel = 0.0;
    for (size_t i = 0; i < actN; ++i) {
        double r = bf16_to_f32(ref_h[i]);
        double t = bf16_to_f32(tw_h[i]);
        double d = std::fabs(r - t);
        max_abs = std::max(max_abs, d);
        sum_abs += d;
        sum_ref += std::fabs(r);
        max_ref = std::max(max_ref, std::fabs(r));
        double denom = std::fabs(r) + 1e-6;
        max_rel = std::max(max_rel, d / denom);
    }
    double mean_abs = sum_abs / actN;
    double mean_ref = sum_ref / actN;
    printf("\nCorrectness (TwELL %s vs fp32 reference)\n",
           args.high_precision ? "fp32" : "bf16");
    printf("  reference   : mean|out|=%.5f  max|out|=%.5f\n", mean_ref, max_ref);
    printf("  abs error   : mean=%.6f  max=%.6f\n", mean_abs, max_abs);
    printf("  rel error   : mean=%.4f%%  max(vs mean|out|)=%.4f%%\n",
           (mean_abs / (mean_ref + 1e-12)) * 100.0,
           (max_abs / (mean_ref + 1e-12)) * 100.0);

    double rel_mean = mean_abs / (mean_ref + 1e-12);
    bool pass = !overflow && rel_mean < 0.03;
    printf("  RESULT      : %s\n", pass ? "PASS" : "FAIL");

    TWELL_D2T::destroy_all_d2t_layers();
    CUDA_CHECK(hipFree(GATE));
    CUDA_CHECK(hipFree(UP));
    CUDA_CHECK(hipFree(DOWN));
    CUDA_CHECK(hipFree(A));
    CUDA_CHECK(hipFree(A_twell));
    CUDA_CHECK(hipFree(OUT_ref));
    CUDA_CHECK(hipFree(OUT_twell));
    CUDA_CHECK(hipFree(H));
    CUDA_CHECK(hipFree(gate_raw));
    CUDA_CHECK(hipFree(up_raw));
    CUDA_CHECK(hipFree(active));
    CUDA_CHECK(hipFree(packed));
    CUDA_CHECK(hipFree(tile_counts));
    return pass ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Performance benchmark
// ---------------------------------------------------------------------------
static int run_benchmark(const Args& args) {
    const size_t M = static_cast<size_t>(args.batch) * args.seq;
    validate_shape(args, M);
    const int hidden = args.hidden;
    const int intermediate = args.intermediate;

    hipDeviceProp_t prop{};
    CUDA_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("Device: %s (%s, %d CUs)\n", prop.name, prop.gcnArchName,
           prop.multiProcessorCount);
    {
        // TODO  support more devices
        const std::string arch(prop.gcnArchName);
        if (arch.find("gfx942") == std::string::npos &&
            arch.find("gfx950") == std::string::npos) {
            fprintf(stderr,
                    "warning: the TwELL D2T kernel uses AMD matrix cores tuned "
                    "for gfx942 (CDNA3) / gfx950 (CDNA4); this device is %s and "
                    "may under-perform or fail to launch.\n",
                    prop.gcnArchName);
        }
    }
    printf(
        "Config: tokens(M)=%zu hidden=%d intermediate=%d layers=%d reps=%d "
        "warmup=%d precision=%s gate_bias=%.5f\n",
        M, hidden, intermediate, args.layers, args.reps, args.warmup,
        args.high_precision ? "fp32" : "bf16", args.gate_bias);

    const size_t wN = static_cast<size_t>(intermediate) * hidden;
    std::vector<at::BFloat16*> gate(args.layers), up(args.layers), down(args.layers);
    for (int l = 0; l < args.layers; ++l) {
        gate[l] = alloc_and_fill(wN, 0x1000u + l, kWeightScale, args.gate_bias);
        up[l] = alloc_and_fill(wN, 0x2000u + l, kWeightScale, 0.0f);
        down[l] = alloc_and_fill(wN, 0x3000u + l, kWeightScale, 0.0f);
    }

    const size_t actN = M * hidden;
    at::BFloat16* act_a = alloc_and_fill(actN, 0xABCDu, kActScale, kActOffset);

    at::BFloat16* act_b = nullptr;
    CUDA_CHECK(hipMalloc(&act_b, actN * sizeof(at::BFloat16)));

    uint32_t* packed = nullptr;
    CUDA_CHECK(hipMalloc(&packed, M * (intermediate / 8) * sizeof(uint32_t)));

    for (int l = 0; l < args.layers; ++l) {
        TWELL_D2T::create_d2t_layer_128x256x64TS8(l, gate[l], hidden, intermediate);
        TWELL_D2T::ensure_d2t_layer_shape_128x256x64TS8(l, M);
    }
    CUDA_CHECK(hipDeviceSynchronize());

    auto run_stack = [&]() {
        at::BFloat16* in = act_a;
        at::BFloat16* out = act_b;
        for (int l = 0; l < args.layers; ++l) {
            if (args.high_precision) {
                TWELL_MLP::run_gated_mlp_layer_128x256x64TS8_high_precision(
                    l, in, up[l], down[l], packed, out, M, intermediate, hidden);
            } else {
                TWELL_MLP::run_gated_mlp_layer_128x256x64TS8(
                    l, in, up[l], down[l], packed, out, M, intermediate, hidden);
            }
            at::BFloat16* tmp = in; in = out; out = tmp;
        }
    };

    printf("Warmup (%d reps)...\n", args.warmup);
    for (int r = 0; r < args.warmup; ++r) run_stack();
    CUDA_CHECK(hipDeviceSynchronize());

    printf("Timing (%d reps)...\n", args.reps);
    auto start = std::chrono::steady_clock::now();

    for (int r = 0; r < args.reps; ++r) run_stack();
    CUDA_CHECK(hipDeviceSynchronize());

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double total_ms = time * 1e-6;

    const double per_rep_ms = total_ms / args.reps;
    const double per_layer_ms = per_rep_ms / args.layers;
    const double tokens_per_s = (M * 1000.0) / per_rep_ms;
    const double dense_flops_per_rep =
        static_cast<double>(M) * intermediate * (6.0 * hidden + 1.0)  * args.layers;
    const double tflops = (dense_flops_per_rep / (per_rep_ms / 1000.0)) / 1e12;

    printf("\nResults\n");
    printf("  total time            : %.3f ms (%d reps)\n", total_ms, args.reps);
    printf("  per forward (stack)   : %.4f ms\n", per_rep_ms);
    printf("  per gated MLP layer   : %.4f ms\n", per_layer_ms);
    printf("  input throughput      : %.1f tokens/s\n", tokens_per_s);
    printf("  dense-equivalent perf : %.2f TFLOP/s\n", tflops);

    TWELL_D2T::destroy_all_d2t_layers();
    for (int l = 0; l < args.layers; ++l) {
        CUDA_CHECK(hipFree(gate[l]));
        CUDA_CHECK(hipFree(up[l]));
        CUDA_CHECK(hipFree(down[l]));
    }
    CUDA_CHECK(hipFree(act_a));
    CUDA_CHECK(hipFree(act_b));
    CUDA_CHECK(hipFree(packed));
    return 0;
}

int main(int argc, char** argv) {
    Args args;
    parse_args(argc, argv, args);
    CUDA_CHECK(hipSetDevice(0));
    if (run_validate(args) == 0)
        run_benchmark(args);
}
