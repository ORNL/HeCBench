// SPDX-License-Identifier: BSD-3-Clause
// Host reference for real spherical harmonics. The values are explicit
// associated-Legendre polynomials in ABACUS order. This translation unit
// does not use either device kernel's recurrence.

#pragma once

#include <cstddef>
#include <vector>

// lmax is 0..12, the range accepted by the benchmark drivers.
class YlmReference {
public:
    explicit YlmReference(int lmax);

    // True when both results match this reference for vectors
    // [first, first + chunk). Component c of local vector j is
    // baseline[c * pitch + j] and optimized[c * pitch + j]. Coordinates of
    // global vector v are host_vectors[3 * v + {0, 1, 2}]. Vector 0 is also
    // compared with the closed form on the positive z axis.
    bool matches_range(const std::vector<double>& host_vectors, size_t first,
                       size_t chunk, const double* baseline,
                       const double* optimized, size_t pitch) const;

private:
    void evaluate(double x, double y, double z,
                  std::vector<double>& harmonics) const;
    bool positive_z_pole(const std::vector<double>& column) const;
    bool matches_vector(size_t vector, double x, double y, double z,
                        const std::vector<double>& baseline,
                        const std::vector<double>& optimized) const;

    int lmax_;
    std::vector<int> order_;
    std::vector<int> trig_;
    std::vector<int> parity_;
    std::vector<size_t> first_;
    std::vector<int> terms_;
    std::vector<double> coefficients_;
};

// Vectors copied to the host per verification step by the device builds.
inline constexpr size_t ylm_verify_chunk_vectors = 16384;

// Host point for the neighbor-list reference. Device ports copy their
// position types into this layout before checking.
struct NeighborSample {
    double x;
    double y;
    double z;
};

// True when both count arrays match every particle. The expected count walks
// every cell that can hold a point inside the cutoff, so a 27-cell stencil
// with too small an edge fails for the particle that owns the missed pair.
// The first 512 particles are also compared with an all-pairs count.
bool neighbor_reference_matches(const NeighborSample* positions, int particle_count,
                                const int* baseline, const int* optimized,
                                double cutoff);
