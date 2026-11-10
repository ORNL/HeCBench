// SPDX-License-Identifier: BSD-3-Clause
// Baseline spherical_harmonics derived from ABACUS cal_ylm_real (LGPL-3.0), commit 2d1da70.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "kernels.h"
#include "reference.h"

// (lmax + 1)^2 real harmonics for each input vector.
static size_t harmonic_elements(const size_t count, const int lmax)
{
    const size_t degree_terms = lmax + 1;
    return degree_terms * degree_terms * count;
}

// One vector per thread. Stores the associated-Legendre triangle, then writes
// the real spherical harmonics in ABACUS order.
static void launch_baseline(const double* __restrict__ vectors,
                            double* __restrict__ legendre,
                            double* __restrict__ harmonics,
                            const size_t count, const int lmax)
{
    #pragma omp target teams distribute parallel for \
        is_device_ptr(vectors, legendre, harmonics) \
        thread_limit(256)
    for (size_t index = 0; index < count; ++index) {
        constexpr double pi = 3.141592653589793238462643383279502884;
        const size_t offset = 3 * index;
        const double x = vectors[offset];
        const double y = vectors[offset + 1];
        const double z = vectors[offset + 2];
        const double magnitude = sqrt(x * x + y * y + z * z);
        const double cos_theta = magnitude < 1.0e-9 ? 0.0 : z / magnitude;
        const double phi = x > 1.0e-9 ? atan(y / x)
                         : x < -1.0e-9 ? atan(y / x) + pi
                         : (y >= 0.0 ? pi / 2.0 : -pi / 2.0);
        const int p_dimension = lmax + 1;
        const size_t stride = count;
        int output = 0;

        for (int l = 0; l <= lmax; ++l) {
            const double normalization = sqrt((2.0 * l + 1.0) / (4.0 * pi));
            if (l == 0) {
                legendre[index] = 1.0;
            } else if (l == 1) {
                legendre[stride + index] = cos_theta;
                const double sin_squared =
                    fmax(0.0, 1.0 - cos_theta * cos_theta);
                legendre[(p_dimension + 1) * stride + index] =
                    -sqrt(sin_squared);
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
                    fmax(0.0, 1.0 - cos_theta * cos_theta);
                double odd_factorial = 1.0;
                for (int value = 2 * l - 1; value > 2; value -= 2)
                    odd_factorial *= value;
                double diagonal =
                    odd_factorial * pow(sin_squared,
                                        static_cast<double>(l) / 2.0);
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
                const double scale = normalization * sqrt(2.0) *
                    sqrt(numerator / denominator);
                const double p =
                    legendre[(m * p_dimension + l) * stride + index];
                harmonics[output++ * stride + index] =
                    scale * p * cos(m * phi);
                harmonics[output++ * stride + index] =
                    scale * p * sin(m * phi);
            }
        }
    }
}

// Computes the same values without materializing the associated-Legendre
// triangle.  Iterating by m needs only the previous two P_l^m values.  The
// normalization and sin(m*phi)/cos(m*phi) terms are also advanced by
// recurrence, removing factorial, pow, atan, and per-output trig calls.
static void launch_optimized(const double* __restrict__ vectors,
                             double* __restrict__ harmonics,
                             const size_t count, const int lmax)
{
    #pragma omp target teams distribute parallel for \
        is_device_ptr(vectors, harmonics) \
        thread_limit(256)
    for (size_t index = 0; index < count; ++index) {
        constexpr double inverse_sqrt_four_pi = 0.28209479177387814347;
        constexpr double sqrt_two = 1.41421356237309504880;
        const size_t offset = 3 * index;
        const double x = vectors[offset];
        const double y = vectors[offset + 1];
        const double z = vectors[offset + 2];
        const double magnitude = sqrt(x * x + y * y + z * z);
        const double cos_theta = magnitude < 1.0e-9 ? 0.0 : z / magnitude;
        const double sin_theta =
            sqrt(fmax(0.0, 1.0 - cos_theta * cos_theta));
        // Same branches as the baseline phi: |x| <= 1e-9 means phi = +-pi/2.
        const bool x_on_axis = fabs(x) <= 1.0e-9;
        const double xy_magnitude = sqrt(x * x + y * y);
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
                norm_mm *= sqrt(static_cast<double>(2 * m + 1) /
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
                        sqrt((static_cast<double>(2 * l + 3) /
                              (2 * l + 1)) *
                             (static_cast<double>(l + 1 - m) /
                              (l + 1 + m)));
                p_lm_minus_two = p_lm_minus_one;
                p_lm_minus_one = p_lm;
            }
        }
    }
}

// Counts outputs that differ by more than 1e-11 + 1e-10*|reference|.
static unsigned long long compare_harmonics(const double* __restrict__ reference,
                                            const double* __restrict__ result,
                                            const size_t size)
{
    unsigned long long mismatches = 0;
    #pragma omp target teams distribute parallel for \
        is_device_ptr(reference, result) \
        reduction(+:mismatches)
    for (size_t index = 0; index < size; ++index) {
        const double difference = fabs(reference[index] - result[index]);
        const double tolerance =
            1.0e-11 + 1.0e-10 * fabs(reference[index]);
        if (!std::isfinite(reference[index]) ||
            !std::isfinite(result[index]) || difference > tolerance)
            ++mismatches;
    }
    return mismatches;
}

// Copies vectors [first, first + chunk) of every component to host, packed as
// host[c * chunk + j]. Each component is one contiguous copy of chunk values.
// The update returns that slice to the mapped allocation; memcpy then packs it.
static void copy_vector_chunk(double* mapped, size_t count, size_t first,
                              size_t chunk, size_t components, double* host)
{
    for (size_t component = 0; component < components; ++component) {
        const size_t offset = component * count + first;
        #pragma omp target update from(mapped[offset:chunk])
        std::memcpy(host + component * chunk, mapped + offset,
                    chunk * sizeof(double));
    }
}

// Checks both kernels against the host reference, then times them.
int run_ylm(size_t count, int lmax, long long iterations)
{
    if (!ylm_launch_ok(count, lmax, iterations)) {
        std::fprintf(stderr, "invalid spherical-harmonic size\n");
        return EXIT_FAILURE;
    }
    const size_t harmonic_count = harmonic_elements(count, lmax);
    const size_t vector_count = count * 3;
    std::vector<double> host_vectors(vector_count);
    for (size_t i = 0; i < count; ++i) {
        host_vectors[3 * i] = -0.75 + (i % 97) * 0.017;
        host_vectors[3 * i + 1] = -0.5 + (i % 71) * 0.017;
        host_vectors[3 * i + 2] = -0.6 + (i % 53) * 0.023;
    }
    // +z axis, the zero vector, and both x == 0 branches. The host reference
    // checks every vector, including these.
    const double edge_vectors[][3] = {
        {0.0, 0.0, 1.0}, {0.0, 0.0, 0.0}, {0.0, -0.5, 0.25}, {0.0, 0.5, -0.25}};
    for (size_t i = 0; i < 4 && i < count; ++i)
        for (int k = 0; k < 3; ++k) host_vectors[3 * i + k] = edge_vectors[i][k];

    std::vector<double> legendre(harmonic_count);
    std::vector<double> harmonics(harmonic_count);
    std::vector<double> optimized_harmonics(harmonic_count);
    double* vectors = host_vectors.data();
    double* legendre_ptr = legendre.data();
    double* harmonics_ptr = harmonics.data();
    double* optimized_ptr = optimized_harmonics.data();

    #pragma omp target enter data \
        map(to: vectors[0:vector_count]) \
        map(alloc: legendre_ptr[0:harmonic_count], \
                   harmonics_ptr[0:harmonic_count], \
                   optimized_ptr[0:harmonic_count])

    // Untimed launches fill the buffers compared with the host reference.
    #pragma omp target data \
        use_device_ptr(vectors, legendre_ptr, harmonics_ptr, optimized_ptr)
    {
        launch_baseline(vectors, legendre_ptr, harmonics_ptr, count, lmax);
        launch_optimized(vectors, optimized_ptr, count, lmax);
    }
    const YlmReference reference(lmax);
    const size_t components =
        static_cast<size_t>(lmax + 1) * static_cast<size_t>(lmax + 1);
    const size_t chunk_capacity =
        count < ylm_verify_chunk_vectors ? count : ylm_verify_chunk_vectors;
    std::vector<double> host_baseline(components * chunk_capacity);
    std::vector<double> host_optimized(components * chunk_capacity);
    bool reference_ok = true;
    for (size_t first = 0; first < count && reference_ok; first += chunk_capacity) {
        const size_t chunk =
            count - first < chunk_capacity ? count - first : chunk_capacity;
        copy_vector_chunk(harmonics_ptr, count, first, chunk, components,
                          host_baseline.data());
        copy_vector_chunk(optimized_ptr, count, first, chunk, components,
                          host_optimized.data());
        reference_ok = reference.matches_range(host_vectors, first, chunk,
                                               host_baseline.data(),
                                               host_optimized.data(), chunk);
    }
    if (!reference_ok) {
        std::printf("ABACUS YLM: FAIL  vectors=%zu lmax=%d verification failed; "
                    "timing skipped\n",
                    count, lmax);
        #pragma omp target exit data \
            map(release: vectors[0:vector_count], \
                         legendre_ptr[0:harmonic_count], \
                         harmonics_ptr[0:harmonic_count], \
                         optimized_ptr[0:harmonic_count])
        return EXIT_FAILURE;
    }

    unsigned long long mismatches = 0;
    double baseline_milliseconds = 0.0;
    double optimized_milliseconds = 0.0;
    #pragma omp target data \
        use_device_ptr(vectors, legendre_ptr, harmonics_ptr, optimized_ptr)
    {
        const auto baseline_start = std::chrono::steady_clock::now();
        for (long long i = 0; i < iterations; ++i)
            launch_baseline(vectors, legendre_ptr, harmonics_ptr, count, lmax);
        baseline_milliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - baseline_start).count();

        const auto optimized_start = std::chrono::steady_clock::now();
        for (long long i = 0; i < iterations; ++i)
            launch_optimized(vectors, optimized_ptr, count, lmax);
        optimized_milliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - optimized_start).count();

        mismatches =
            compare_harmonics(harmonics_ptr, optimized_ptr, harmonic_count);
    }

    const bool valid = mismatches == 0;
    const double baseline_time = baseline_milliseconds / iterations;
    const double optimized_time = optimized_milliseconds / iterations;
    std::printf("ABACUS YLM: %s  vectors=%zu lmax=%d baseline=%.4f ms "
                "optimized=%.4f ms speedup=%.2fx mismatches=%llu\n",
                valid ? "PASS" : "FAIL", count, lmax, baseline_time,
                optimized_time, baseline_time / optimized_time, mismatches);

    #pragma omp target exit data \
        map(release: vectors[0:vector_count], legendre_ptr[0:harmonic_count], \
                     harmonics_ptr[0:harmonic_count], \
                     optimized_ptr[0:harmonic_count])
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
