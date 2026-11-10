#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <hip/hip_runtime.h>

#include "launch.h"
#include "reference.h"

// Abort with the HIP error string when a runtime call fails.
#define CHECK(call)                                                                          \
    do {                                                                                     \
        const hipError_t err_ = (call);                                                      \
        if (err_ != hipSuccess) {                                                            \
            fprintf(stderr, "HIP error %s:%d '%s': %s\n", __FILE__, __LINE__, #call,        \
                    hipGetErrorString(err_));                                                \
            exit(EXIT_FAILURE);                                                              \
        }                                                                                    \
    } while (0)

// xorshift32 generator used to build the synthetic scene.
struct Rng {
    uint32_t state;
    // Seed the generator. The seed is the first state, so 0 stays 0.
    explicit Rng(uint32_t seed) : state(seed) {}
    // Advance one step and return the new state.
    uint32_t next() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    }
    // Float in [0, 1) taken from the top 24 bits of the state.
    float uniform() { return (next() >> 8) * (1.0f / 16777216.0f); }
    // Standard normal sample from the Box-Muller transform. u1 is kept away
    // from 0 so the logarithm stays finite.
    float normal() {
        float u1 = std::max(uniform(), 1.0e-7f);
        float u2 = uniform();
        return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.28318530718f * u2);
    }
};

struct Scene {
    std::vector<vec2> means2d;
    std::vector<vec3> conics;
    std::vector<float> colors;
    std::vector<float> opacities;
    std::vector<int> tile_offsets;
    std::vector<int> flatten_ids;
    std::vector<float> v_render_colors;
    std::vector<float> v_render_alphas;
    int n = 0;
    int width = 0;
    int height = 0;
    int tile_w = 0;
    int tile_h = 0;
    int busiest = 0;
};

// One synthetic scene. Three quarters of the Gaussians cluster at the image
// center and the rest are placed uniformly. Each ellipse has a cross term and
// its inverse covariance stays positive definite. The nearest 5% are snapped
// to a pixel center with opacity 1, so that pixel hits the 0.999 alpha clamp.
// Each Gaussian is inserted into every 16x16 tile its 3-sigma footprint
// overlaps. Intersections are sorted by tile, then near to far. The incoming
// pixel color and alpha gradients are drawn uniformly from [-1, 1].
static Scene make_scene(int n, int height, int width, int seed) {
    Scene scene;
    scene.n = n;
    scene.width = width;
    scene.height = height;
    scene.tile_w = (width + kTileSize - 1) / kTileSize;
    scene.tile_h = (height + kTileSize - 1) / kTileSize;
    const int tiles = scene.tile_w * scene.tile_h;

    scene.means2d.resize(n);
    scene.conics.resize(n);
    scene.colors.resize((size_t)n * kCdim);
    scene.opacities.resize(n);

    Rng rng(seed);
    bool saw_clamp = false;
    bool saw_cross = false;
    struct Isect {
        int tile;
        float depth;
        int id;
    };
    std::vector<Isect> isects;
    isects.reserve((size_t)n * 4);

    for (int g = 0; g < n; ++g) {
        const float depth = rng.uniform();
        float x, y;
        if (rng.uniform() < 0.75f) {
            x = 0.5f * width + 0.08f * width * rng.normal();
            y = 0.5f * height + 0.08f * height * rng.normal();
        } else {
            x = rng.uniform() * width;
            y = rng.uniform() * height;
        }
        const float sx = 1.5f + 4.5f * rng.uniform();
        const float sy = 1.5f + 4.5f * rng.uniform();
        const float cxx = 1.0f / (sx * sx);
        const float cyy = 1.0f / (sy * sy);
        // |cxy| stays below half of sqrt(cxx * cyy), so cxx * cyy - cxy^2 > 0.
        const float cxy = 0.5f * std::sqrt(cxx * cyy) * (2.0f * rng.uniform() - 1.0f);
        const bool clamped = depth < 0.05f;
        if (clamped) {
            x = std::min(std::max(std::floor(x) + 0.5f, 0.5f), width - 0.5f);
            y = std::min(std::max(std::floor(y) + 0.5f, 0.5f), height - 0.5f);
        }
        scene.means2d[g] = {x, y};
        scene.conics[g] = {cxx, cxy, cyy};
        scene.colors[(size_t)g * 3 + 0] = rng.uniform();
        scene.colors[(size_t)g * 3 + 1] = rng.uniform();
        scene.colors[(size_t)g * 3 + 2] = rng.uniform();
        scene.opacities[g] = clamped ? 1.0f : (0.05f + 0.20f * rng.uniform());
        saw_clamp = saw_clamp || clamped;
        saw_cross = saw_cross || cxy != 0.0f;

        const float det = cxx * cyy - cxy * cxy;
        const float radx = 3.0f * std::sqrt(cyy / det);
        const float rady = 3.0f * std::sqrt(cxx / det);
        int tx0 = static_cast<int>(std::floor((x - radx) / kTileSize));
        int tx1 = static_cast<int>(std::floor((x + radx) / kTileSize));
        int ty0 = static_cast<int>(std::floor((y - rady) / kTileSize));
        int ty1 = static_cast<int>(std::floor((y + rady) / kTileSize));
        tx0 = std::max(tx0, 0);
        ty0 = std::max(ty0, 0);
        tx1 = std::min(tx1, scene.tile_w - 1);
        ty1 = std::min(ty1, scene.tile_h - 1);
        for (int ty = ty0; ty <= ty1; ++ty) {
            for (int tx = tx0; tx <= tx1; ++tx) {
                isects.push_back({ty * scene.tile_w + tx, depth, g});
            }
        }
    }

    std::sort(isects.begin(), isects.end(), [](const Isect& a, const Isect& b) {
        if (a.tile != b.tile) return a.tile < b.tile;
        return a.depth < b.depth;
    });

    scene.tile_offsets.assign(tiles, 0);
    scene.flatten_ids.resize(isects.size());
    std::vector<int> counts(tiles, 0);
    for (const Isect& item : isects) counts[item.tile] += 1;
    scene.busiest = 0;
    for (int tile = 0; tile < tiles; ++tile) {
        scene.busiest = std::max(scene.busiest, counts[tile]);
        if (tile > 0) scene.tile_offsets[tile] = scene.tile_offsets[tile - 1] + counts[tile - 1];
    }
    for (size_t i = 0; i < isects.size(); ++i) scene.flatten_ids[i] = isects[i].id;

    const size_t pixels = (size_t)width * height;
    scene.v_render_colors.resize(pixels * kCdim);
    scene.v_render_alphas.resize(pixels);
    Rng grad(seed + 1000);
    for (size_t i = 0; i < scene.v_render_colors.size(); ++i) {
        scene.v_render_colors[i] = grad.uniform() * 2.0f - 1.0f;
    }
    for (size_t i = 0; i < scene.v_render_alphas.size(); ++i) {
        scene.v_render_alphas[i] = grad.uniform() * 2.0f - 1.0f;
    }
    if (!saw_clamp || !saw_cross) {
        printf("Error: the scene never hits the alpha clamp or the conic cross term\n");
        exit(EXIT_FAILURE);
    }
    return scene;
}

// Allocate a device buffer and copy the host vector into it.
template <typename T>
static T* device_copy(const std::vector<T>& host) {
    T* ptr = nullptr;
    CHECK(hipMalloc(&ptr, host.size() * sizeof(T)));
    if (!host.empty()) {
        CHECK(hipMemcpy(ptr, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice));
    }
    return ptr;
}

// Clear the four gradient buffers. The backward kernels atomicAdd into them,
// so they have to start at zero. The clear stays outside the timed interval.
static void zero_grads(BwdArgs& args) {
    CHECK(hipMemset(args.v_means2d, 0, args.n * sizeof(vec2)));
    CHECK(hipMemset(args.v_conics, 0, args.n * sizeof(vec3)));
    CHECK(hipMemset(args.v_colors, 0, (size_t)args.n * kCdim * sizeof(float)));
    CHECK(hipMemset(args.v_opacities, 0, args.n * sizeof(float)));
}

struct Grads {
    std::vector<float> means2d;
    std::vector<float> conics;
    std::vector<float> colors;
    std::vector<float> opacities;
};

// Copy the four device gradient buffers back to the host.
// Layout is means2d n*2, conics n*3, colors n*3, opacities n.
static Grads download(const BwdArgs& args) {
    Grads g;
    g.means2d.resize((size_t)args.n * 2);
    g.conics.resize((size_t)args.n * 3);
    g.colors.resize((size_t)args.n * kCdim);
    g.opacities.resize(args.n);
    CHECK(hipMemcpy(g.means2d.data(), args.v_means2d, g.means2d.size() * sizeof(float), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(g.conics.data(), args.v_conics, g.conics.size() * sizeof(float), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(g.colors.data(), args.v_colors, g.colors.size() * sizeof(float), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(g.opacities.data(), args.v_opacities, g.opacities.size() * sizeof(float), hipMemcpyDeviceToHost));
    return g;
}

// True when both values are finite and |got - ref| <= tol * (1 + |ref|).
static bool close_enough(float got, float ref, float tol) {
    if (!std::isfinite(got) || !std::isfinite(ref)) return false;
    const float diff = std::fabs(got - ref);
    return diff <= tol + tol * std::fabs(ref);
}

// Compare one buffer with the reference. Prints at most four mismatches and
// returns how many elements failed. *max_abs receives the largest |got - ref|.
static int compare_buffer(const char* tag, const char* name, const std::vector<float>& got,
                          const std::vector<float>& ref, float tol, float* max_abs) {
    int errors = 0;
    float local_max = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float diff = std::fabs(got[i] - ref[i]);
        local_max = std::max(local_max, diff);
        if (!close_enough(got[i], ref[i], tol)) {
            if (errors < 4) {
                printf("  %s %s[%zu] device %.8g reference %.8g\n", tag, name, i, got[i], ref[i]);
            }
            ++errors;
        }
    }
    *max_abs = local_max;
    return errors;
}

// Compare all four gradient buffers with the host reference.
// The device rebuilds transmittance as 1 - render_alpha and accumulates with
// float atomics. The host keeps the forward product and sums in double. On
// the four study scenes, over three runs each on H100 (CUDA, SYCL) and
// MI210 (HIP, SYCL), the largest diff/(1+|ref|) was 2.3e-4 for means, 8.6e-3
// for conics, 1.2e-3 for colors, and 8.6e-4 for opacities. Each tolerance
// below is about twice that. out_max receives the four max |diff|.
static int compare_grads(const char* tag, const Grads& got, const Reference& ref, float out_max[4]) {
    constexpr float kMeansTol = 5.0e-4f;
    constexpr float kConicTol = 2.0e-2f;
    constexpr float kColorTol = 2.5e-3f;
    constexpr float kOpacTol = 2.0e-3f;
    return compare_buffer(tag, "v_means2d", got.means2d, ref.means2d, kMeansTol, &out_max[0]) +
           compare_buffer(tag, "v_conics", got.conics, ref.conics, kConicTol, &out_max[1]) +
           compare_buffer(tag, "v_colors", got.colors, ref.colors, kColorTol, &out_max[2]) +
           compare_buffer(tag, "v_opacities", got.opacities, ref.opacities, kOpacTol, &out_max[3]);
}

// Sum of every reference gradient. A zero sum means the scene produced no
// training signal, and the check fails.
static double checksum(const Reference& g) {
    double sum = 0.0;
    for (float v : g.means2d) sum += v;
    for (float v : g.conics) sum += v;
    for (float v : g.colors) sum += v;
    for (float v : g.opacities) sum += v;
    return sum;
}

// Average time of one backward launch, in milliseconds. Warmup launches are
// discarded. zero_grads runs before the timed region.
static float time_kernel(void (*launch)(const BwdArgs&), BwdArgs& args, int warmup, int repeat) {
    for (int i = 0; i < warmup; ++i) {
        zero_grads(args);
        launch(args);
    }
    CHECK(hipDeviceSynchronize());
    double total_ms = 0.0;
    for (int i = 0; i < repeat; ++i) {
        zero_grads(args);
        CHECK(hipDeviceSynchronize());
        const auto start = std::chrono::steady_clock::now();
        launch(args);
        CHECK(hipDeviceSynchronize());
        const auto stop = std::chrono::steady_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(stop - start).count();
    }
    CHECK(hipGetLastError());
    return static_cast<float>(total_ms / repeat);
}

// Parse a decimal integer in [1, max]. Returns 0 when the text is not one.
static long parse_count(const char* text, long max) {
    char* end = nullptr;
    errno = 0;
    const long value = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value < 1 || value > max) return 0;
    return value;
}

// Build one scene, check the forward pass and both backward kernels against
// the host reference, then print the average launch time of each kernel.
int main(int argc, char** argv) {
    if (argc != 5) {
        printf("Usage: %s <gaussians> <height> <width> <repeat>\n", argv[0]);
        return EXIT_FAILURE;
    }
    // Intersection ids and pixel ids are 32-bit on the device.
    const int n = (int)parse_count(argv[1], 1L << 26);
    const int height = (int)parse_count(argv[2], 1L << 14);
    const int width = (int)parse_count(argv[3], 1L << 14);
    const int repeat = (int)parse_count(argv[4], 1L << 20);
    if (n == 0 || height == 0 || width == 0 || repeat == 0) {
        printf("Error: gaussians must be in [1, %ld], height and width in [1, %ld], repeat in [1, %ld]\n",
               1L << 26, 1L << 14, 1L << 20);
        return EXIT_FAILURE;
    }

    const Scene scene = make_scene(n, height, width, 42);
    if (scene.flatten_ids.empty()) {
        printf("Error: the scene produced no tile intersections\n");
        return EXIT_FAILURE;
    }
    // Kernels add up to one block of padding to an id range, so keep that
    // headroom below INT32_MAX.
    if (scene.flatten_ids.size() > (size_t)INT32_MAX - kTileSize * kTileSize) {
        printf("Error: %zu tile intersections do not fit 32-bit ids\n", scene.flatten_ids.size());
        return EXIT_FAILURE;
    }
    printf("Gaussians: %d, image: %d x %d (%d tiles), intersections: %zu (busiest tile %d)\n", n, height,
           width, scene.tile_w * scene.tile_h, scene.flatten_ids.size(), scene.busiest);

    const size_t pixels = (size_t)width * height;
    BwdArgs base{};
    base.n = n;
    base.n_isects = (int)scene.flatten_ids.size();
    base.image_width = width;
    base.image_height = height;
    base.tile_width = scene.tile_w;
    base.tile_height = scene.tile_h;
    base.means2d = device_copy(scene.means2d);
    base.conics = device_copy(scene.conics);
    base.colors = device_copy(scene.colors);
    base.opacities = device_copy(scene.opacities);
    base.tile_offsets = device_copy(scene.tile_offsets);
    base.flatten_ids = device_copy(scene.flatten_ids);
    base.v_render_colors = device_copy(scene.v_render_colors);
    base.v_render_alphas = device_copy(scene.v_render_alphas);
    CHECK(hipMalloc(const_cast<float**>(&base.render_alphas), pixels * sizeof(float)));
    CHECK(hipMalloc(const_cast<int**>(&base.last_ids), pixels * sizeof(int)));
    CHECK(hipMemset(const_cast<float*>(base.render_alphas), 0, pixels * sizeof(float)));
    CHECK(hipMemset(const_cast<int*>(base.last_ids), 0, pixels * sizeof(int)));

    BwdArgs opt = base;
    CHECK(hipMalloc(&opt.v_means2d, n * sizeof(vec2)));
    CHECK(hipMalloc(&opt.v_conics, n * sizeof(vec3)));
    CHECK(hipMalloc(&opt.v_colors, (size_t)n * kCdim * sizeof(float)));
    CHECK(hipMalloc(&opt.v_opacities, n * sizeof(float)));
    CHECK(hipMalloc(&base.v_means2d, n * sizeof(vec2)));
    CHECK(hipMalloc(&base.v_conics, n * sizeof(vec3)));
    CHECK(hipMalloc(&base.v_colors, (size_t)n * kCdim * sizeof(float)));
    CHECK(hipMalloc(&base.v_opacities, n * sizeof(float)));

    launch_rasterize_forward(base);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());

    zero_grads(base);
    launch_rasterize_baseline(base);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    zero_grads(opt);
    launch_rasterize_optimized(opt);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());

    const Reference ref = reference_rasterize(n, width, height, scene.tile_w, scene.tile_h, scene.means2d,
                                              scene.conics, scene.colors, scene.opacities, scene.tile_offsets,
                                              scene.flatten_ids, scene.v_render_colors, scene.v_render_alphas);
    std::vector<float> gpu_alphas(pixels);
    std::vector<int> gpu_last(pixels);
    CHECK(hipMemcpy(gpu_alphas.data(), base.render_alphas, pixels * sizeof(float), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(gpu_last.data(), base.last_ids, pixels * sizeof(int), hipMemcpyDeviceToHost));

    float max_alpha = 0.0f;
    // Forward alpha and last_ids are produced by the same rounded evaluation as
    // the host reference. The tolerance covers a platform that flushes a
    // subnormal the host keeps.
    const int alpha_errors =
        compare_buffer("forward", "render_alphas", gpu_alphas, ref.render_alphas, 2.5e-6f, &max_alpha);
    int last_errors = 0;
    for (size_t i = 0; i < pixels; ++i) {
        if (gpu_last[i] != ref.last_ids[i]) {
            if (last_errors < 4) {
                printf("  forward last_ids[%zu] device %d reference %d\n", i, gpu_last[i], ref.last_ids[i]);
            }
            ++last_errors;
        }
    }

    const Grads baseline = download(base);
    const Grads optimized = download(opt);
    float base_max[4] = {};
    float opt_max[4] = {};
    const int base_errors = compare_grads("baseline", baseline, ref, base_max);
    const int opt_errors = compare_grads("optimized", optimized, ref, opt_max);
    const double grad_sum = checksum(ref);
    const bool nonzero = grad_sum != 0.0;
    const bool pass = alpha_errors == 0 && last_errors == 0 && base_errors == 0 && opt_errors == 0 && nonzero;
    printf("Rasterize backward: %s\n", pass ? "PASS" : "FAIL");
    printf("  forward   max abs error  alpha %.3e  last_id mismatches %d\n", max_alpha, last_errors);
    printf("  baseline  max abs error  means2d %.3e  conics %.3e  colors %.3e  opacities %.3e\n", base_max[0],
           base_max[1], base_max[2], base_max[3]);
    printf("  optimized max abs error  means2d %.3e  conics %.3e  colors %.3e  opacities %.3e\n", opt_max[0],
           opt_max[1], opt_max[2], opt_max[3]);
    printf("Checksum (reference gradient sum): %.8g\n", grad_sum);
    if (!nonzero) {
        printf("Error: the reference gradients are all zero\n");
    }

    const int warmup = 100;
    const float base_ms = time_kernel(launch_rasterize_baseline, base, warmup, repeat);
    const float opt_ms = time_kernel(launch_rasterize_optimized, opt, warmup, repeat);
    printf("Average execution time of the baseline kernel: %.3f (us)\n", base_ms * 1.0e3);
    printf("Average execution time of the optimized kernel: %.3f (us)\n", opt_ms * 1.0e3);
    printf("Speedup: %.3fx\n", opt_ms > 0.0f ? base_ms / opt_ms : 0.0);

    // base and opt share every input buffer; only the gradients are separate.
    CHECK(hipFree(const_cast<vec2*>(base.means2d)));
    CHECK(hipFree(const_cast<vec3*>(base.conics)));
    CHECK(hipFree(const_cast<float*>(base.colors)));
    CHECK(hipFree(const_cast<float*>(base.opacities)));
    CHECK(hipFree(const_cast<int*>(base.tile_offsets)));
    CHECK(hipFree(const_cast<int*>(base.flatten_ids)));
    CHECK(hipFree(const_cast<float*>(base.v_render_colors)));
    CHECK(hipFree(const_cast<float*>(base.v_render_alphas)));
    CHECK(hipFree(const_cast<float*>(base.render_alphas)));
    CHECK(hipFree(const_cast<int*>(base.last_ids)));
    for (BwdArgs* args : {&base, &opt}) {
        CHECK(hipFree(args->v_means2d));
        CHECK(hipFree(args->v_conics));
        CHECK(hipFree(args->v_colors));
        CHECK(hipFree(args->v_opacities));
    }

    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
