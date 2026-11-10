// SPDX-License-Identifier: BSD-3-Clause
// Baseline spherical_harmonics derived from ABACUS cal_ylm_real (LGPL-3.0), commit 2d1da70.

#include <chrono>
#include <new>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <sycl/sycl.hpp>

#include "kernels.h"
#include "reference.h"

// One vector per work-item. Stores the associated-Legendre triangle, then
// writes the real spherical harmonics in ABACUS order.
static void launch_baseline(sycl::queue& q, const double* vectors_arg,
                            double* legendre_arg, double* harmonics_arg,
                            const size_t count, const int lmax)
{
    const size_t global = (count + 255) / 256 * 256;
    q.parallel_for(
        sycl::nd_range<1>(global, 256), [=](sycl::nd_item<1> item) {
        const double* __restrict__ vectors = vectors_arg;
        double* __restrict__ legendre = legendre_arg;
        double* __restrict__ harmonics = harmonics_arg;
        const size_t index = item.get_global_id(0);
        if (index >= count) return;

        constexpr double pi = 3.141592653589793238462643383279502884;
        const size_t offset = 3 * index;
        const double x = vectors[offset];
        const double y = vectors[offset + 1];
        const double z = vectors[offset + 2];
        const double magnitude = sycl::sqrt(x * x + y * y + z * z);
        const double cos_theta = magnitude < 1.0e-9 ? 0.0 : z / magnitude;
        const double phi = x > 1.0e-9 ? sycl::atan(y / x)
                         : x < -1.0e-9 ? sycl::atan(y / x) + pi
                         : (y >= 0.0 ? pi / 2.0 : -pi / 2.0);
        const int p_dimension = lmax + 1;
        const size_t stride = count;
        int output = 0;

        for (int l = 0; l <= lmax; ++l) {
            const double normalization =
                sycl::sqrt((2.0 * l + 1.0) / (4.0 * pi));
            if (l == 0) {
                legendre[index] = 1.0;
            } else if (l == 1) {
                legendre[stride + index] = cos_theta;
                const double sin_squared =
                    sycl::fmax(0.0, 1.0 - cos_theta * cos_theta);
                legendre[(p_dimension + 1) * stride + index] =
                    -sycl::sqrt(sin_squared);
            } else {
                for (int m = 0; m <= l - 2; ++m) {
                    legendre[(m * p_dimension + l) * stride + index] =
                        ((2 * l - 1) * cos_theta *
                             legendre[(m * p_dimension + l - 1) * stride + index] -
                         (l - 1 + m) *
                             legendre[(m * p_dimension + l - 2) * stride + index]) /
                        (l - m);
                }
                legendre[((l - 1) * p_dimension + l) * stride + index] =
                    (2 * l - 1) * cos_theta *
                    legendre[((l - 1) * p_dimension + l - 1) * stride + index];
                const double sin_squared =
                    sycl::fmax(0.0, 1.0 - cos_theta * cos_theta);
                double odd_factorial = 1.0;
                for (int value = 2 * l - 1; value > 2; value -= 2)
                    odd_factorial *= value;
                double diagonal =
                    odd_factorial *
                    sycl::pow(sin_squared, static_cast<double>(l) / 2.0);
                legendre[(l * p_dimension + l) * stride + index] =
                    l & 1 ? -diagonal : diagonal;
            }

            harmonics[output++ * stride + index] =
                normalization * legendre[l * stride + index];
            for (int m = 1; m <= l; ++m) {
                double numerator = 1.0, denominator = 1.0;
                for (int value = 2; value <= l - m; ++value)
                    numerator *= value;
                for (int value = 2; value <= l + m; ++value)
                    denominator *= value;
                const double scale = normalization * sycl::sqrt(2.0) *
                    sycl::sqrt(numerator / denominator);
                const double p =
                    legendre[(m * p_dimension + l) * stride + index];
                harmonics[output++ * stride + index] =
                    scale * p * sycl::cos(m * phi);
                harmonics[output++ * stride + index] =
                    scale * p * sycl::sin(m * phi);
            }
        }
    });
}

// Computes the same values without materializing the associated-Legendre
// triangle.  Iterating by m needs only the previous two P_l^m values.  The
// normalization and sin(m*phi)/cos(m*phi) terms are also advanced by
// recurrence, removing factorial, pow, atan, and per-output trig calls.
static void launch_optimized(sycl::queue& q, const double* vectors_arg,
                             double* harmonics_arg, const size_t count,
                             const int lmax)
{
    const size_t global = (count + 255) / 256 * 256;
    q.parallel_for(
        sycl::nd_range<1>(global, 256), [=](sycl::nd_item<1> item) {
        const double* __restrict__ vectors = vectors_arg;
        double* __restrict__ harmonics = harmonics_arg;
        const size_t index = item.get_global_id(0);
        if (index >= count) return;

        constexpr double inverse_sqrt_four_pi = 0.28209479177387814347;
        constexpr double sqrt_two = 1.41421356237309504880;
        const size_t offset = 3 * index;
        const double x = vectors[offset];
        const double y = vectors[offset + 1];
        const double z = vectors[offset + 2];
        const double magnitude = sycl::sqrt(x * x + y * y + z * z);
        const double cos_theta = magnitude < 1.0e-9 ? 0.0 : z / magnitude;
        const double sin_theta =
            sycl::sqrt(sycl::fmax(0.0, 1.0 - cos_theta * cos_theta));
        // Same branches as the baseline phi: |x| <= 1e-9 means phi = +-pi/2.
        const bool x_on_axis = sycl::fabs(x) <= 1.0e-9;
        const double xy_magnitude = sycl::sqrt(x * x + y * y);
        const double cos_phi = x_on_axis ? 0.0 : x / xy_magnitude;
        const double sin_phi = x_on_axis ? (y >= 0.0 ? 1.0 : -1.0)
                                         : y / xy_magnitude;
        const size_t stride = count;

        double p_mm = 1.0;
        double norm_mm = inverse_sqrt_four_pi;
        double cos_m_phi = 1.0;
        double sin_m_phi = 0.0;

        for (int m = 0; m <= lmax; ++m) {
            if (m > 0) {
                p_mm *= -(2 * m - 1) * sin_theta;
                norm_mm *= sycl::sqrt(static_cast<double>(2 * m + 1) /
                                (2.0 * m * (2 * m - 1) * (2 * m - 1)));
                const double next_cos =
                    cos_m_phi * cos_phi - sin_m_phi * sin_phi;
                sin_m_phi = sin_m_phi * cos_phi + cos_m_phi * sin_phi;
                cos_m_phi = next_cos;
            }

            double p_lm_minus_two = 0.0;
            double p_lm_minus_one = 0.0;
            double norm_lm = norm_mm;
            for (int l = m; l <= lmax; ++l) {
                double p_lm;
                if (l == m)
                    p_lm = p_mm;
                else if (l == m + 1)
                    p_lm = (2 * m + 1) * cos_theta * p_mm;
                else
                    p_lm = ((2 * l - 1) * cos_theta * p_lm_minus_one -
                            (l + m - 1) * p_lm_minus_two) / (l - m);

                const size_t base = static_cast<size_t>(l) * l;
                if (m == 0) {
                    harmonics[base * stride + index] = norm_lm * p_lm;
                } else {
                    const double value = sqrt_two * norm_lm * p_lm;
                    harmonics[(base + 2 * m - 1) * stride + index] =
                        value * cos_m_phi;
                    harmonics[(base + 2 * m) * stride + index] =
                        value * sin_m_phi;
                }
                if (l < lmax)
                    norm_lm *=
                        sycl::sqrt((static_cast<double>(2 * l + 3) /
                                    (2 * l + 1)) *
                                   (static_cast<double>(l + 1 - m) /
                                    (l + 1 + m)));
                p_lm_minus_two = p_lm_minus_one;
                p_lm_minus_one = p_lm;
            }
        }
    });
}

// Releases one device allocation. Errors are ignored so a destructor cannot throw.
static void release_device(void* pointer, sycl::queue& q) noexcept
{
    if (pointer == nullptr) return;
    try {
        sycl::free(pointer, q);
    } catch (...) {
    }
}

// Frees every device allocation, including when a later SYCL call throws.
// Pending work is drained first so no kernel still uses freed memory.
struct DeviceMemory {
    sycl::queue& q;
    double* vectors = nullptr;
    double* legendre = nullptr;
    double* harmonics = nullptr;
    double* optimized_harmonics = nullptr;
    unsigned int* mismatches = nullptr;
    explicit DeviceMemory(sycl::queue& q_in) : q(q_in) {}
    ~DeviceMemory() {
        try {
            q.wait();
        } catch (...) {
        }
        release_device(mismatches, q);
        release_device(optimized_harmonics, q);
        release_device(harmonics, q);
        release_device(legendre, q);
        release_device(vectors, q);
    }
    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;
};

// Copies vectors [first, first + chunk) of every component to host, packed as
// host[c * chunk + j]. Each component is one contiguous copy of chunk values.
static void copy_vector_chunk(sycl::queue& q, const double* device,
                              size_t count, size_t first, size_t chunk,
                              size_t components, double* host)
{
    for (size_t component = 0; component < components; ++component) {
        q.memcpy(host + component * chunk,
                     device + component * count + first,
                     chunk * sizeof(double));
    }
    q.wait_and_throw();
}

// Checks both kernels against the host reference, then times them.
int run_ylm(size_t count, int lmax, long long iterations)
{
    if (!ylm_launch_ok(count, lmax, iterations)) {
        std::fprintf(stderr, "invalid spherical-harmonic size\n");
        return EXIT_FAILURE;
    }

    try {
#ifdef USE_GPU
        sycl::queue q{sycl::gpu_selector_v, {sycl::property::queue::in_order{}}};
#else
        sycl::queue q{sycl::cpu_selector_v, {sycl::property::queue::in_order{}}};
#endif
        if (!q.get_device().has(sycl::aspect::fp64)) {
            std::fprintf(stderr, "selected SYCL device lacks fp64 support\n");
            return EXIT_FAILURE;
        }

        const size_t degree_terms = lmax + 1;
        const size_t harmonic_count = degree_terms * degree_terms * count;
        std::vector<double> host_vectors(count * 3);
        for (size_t i = 0; i < count; ++i) {
            host_vectors[3 * i] = -0.75 + (i % 97) * 0.017;
            host_vectors[3 * i + 1] = -0.5 + (i % 71) * 0.017;
            host_vectors[3 * i + 2] = -0.6 + (i % 53) * 0.023;
        }
        // +z axis, the zero vector, and both x == 0 branches. The host reference
        // checks every vector, including these.
        const double edge_vectors[][3] = {{0.0, 0.0, 1.0}, {0.0, 0.0, 0.0},
                                          {0.0, -0.5, 0.25}, {0.0, 0.5, -0.25}};
        for (size_t i = 0; i < 4 && i < count; ++i)
            for (int k = 0; k < 3; ++k)
                host_vectors[3 * i + k] = edge_vectors[i][k];

        DeviceMemory device{q};
        device.vectors = sycl::malloc_device<double>(host_vectors.size(), q);
        device.legendre = sycl::malloc_device<double>(harmonic_count, q);
        device.harmonics = sycl::malloc_device<double>(harmonic_count, q);
        device.optimized_harmonics = sycl::malloc_device<double>(harmonic_count, q);
        device.mismatches = sycl::malloc_device<unsigned int>(1, q);
        if (!device.vectors || !device.legendre || !device.harmonics ||
            !device.optimized_harmonics || !device.mismatches) {
            throw std::bad_alloc();
        }
        double* vectors = device.vectors;
        double* legendre = device.legendre;
        double* harmonics = device.harmonics;
        double* optimized_harmonics = device.optimized_harmonics;
        unsigned int* mismatches = device.mismatches;

        q.memcpy(vectors, host_vectors.data(),
                     host_vectors.size() * sizeof(double)).wait();
        // Untimed launches fill the buffers compared with the host reference.
        launch_baseline(q, vectors, legendre, harmonics, count, lmax);
        launch_optimized(q, vectors, optimized_harmonics, count, lmax);
        q.wait_and_throw();

        const YlmReference reference(lmax);
        const size_t components = degree_terms * degree_terms;
        const size_t chunk_capacity =
            count < ylm_verify_chunk_vectors ? count : ylm_verify_chunk_vectors;
        std::vector<double> host_baseline(components * chunk_capacity);
        std::vector<double> host_optimized(components * chunk_capacity);
        bool reference_ok = true;
        for (size_t first = 0; first < count && reference_ok;
             first += chunk_capacity) {
            const size_t chunk = count - first < chunk_capacity
                                     ? count - first
                                     : chunk_capacity;
            copy_vector_chunk(q, harmonics, count, first, chunk, components,
                              host_baseline.data());
            copy_vector_chunk(q, optimized_harmonics, count, first, chunk,
                              components, host_optimized.data());
            reference_ok = reference.matches_range(
                host_vectors, first, chunk, host_baseline.data(),
                host_optimized.data(), chunk);
        }
        if (!reference_ok) {
            std::printf("ABACUS YLM: FAIL  vectors=%zu lmax=%d verification "
                        "failed; timing skipped\n",
                        count, lmax);
            return EXIT_FAILURE;
        }

        const auto baseline_start = std::chrono::steady_clock::now();
        for (long long i = 0; i < iterations; ++i)
            launch_baseline(q, vectors, legendre, harmonics, count, lmax);
        q.wait();
        const double baseline_milliseconds =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - baseline_start).count();

        const auto optimized_start = std::chrono::steady_clock::now();
        for (long long i = 0; i < iterations; ++i)
            launch_optimized(q, vectors, optimized_harmonics, count, lmax);
        q.wait();
        const double optimized_milliseconds =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - optimized_start).count();

        q.memset(mismatches, 0, sizeof(unsigned int));
        // Counts outputs that differ by more than 1e-11 + 1e-10*|reference|.
        q.parallel_for(sycl::range<1>(harmonic_count), [=](sycl::id<1> id) {
            const double* __restrict__ reference = harmonics;
            const double* __restrict__ result = optimized_harmonics;
            unsigned int* __restrict__ mismatch_count = mismatches;
            const size_t index = id[0];
            const double difference =
                sycl::fabs(reference[index] - result[index]);
            const double tolerance =
                1.0e-11 + 1.0e-10 * sycl::fabs(reference[index]);
            if (!sycl::isfinite(reference[index]) ||
                !sycl::isfinite(result[index]) || difference > tolerance) {
                sycl::atomic_ref<unsigned int, sycl::memory_order::relaxed,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space> counter(
                        mismatch_count[0]);
                counter.fetch_add(1);
            }
        });

        unsigned int host_mismatches = 0;
        q.memcpy(&host_mismatches, mismatches, sizeof(unsigned int));
        q.wait_and_throw();

        const bool valid = host_mismatches == 0;
        const double baseline_time = baseline_milliseconds / iterations;
        const double optimized_time = optimized_milliseconds / iterations;
        std::printf("ABACUS YLM: %s  vectors=%zu lmax=%d baseline=%.4f ms "
                    "optimized=%.4f ms speedup=%.2fx mismatches=%u\n",
                    valid ? "PASS" : "FAIL", count, lmax, baseline_time,
                    optimized_time, baseline_time / optimized_time,
                    host_mismatches);
        return valid ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const sycl::exception& error) {
        std::fprintf(stderr, "SYCL error: %s\n", error.what());
        return EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return EXIT_FAILURE;
    }
}
