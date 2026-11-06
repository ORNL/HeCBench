// SPDX-License-Identifier: BSD-3-Clause
// Entry points for the ABACUS spherical-harmonic and neighbor-list kernels
// hosted by this benchmark.

#pragma once

#include <climits>
#include <cstddef>
#include <cstdint>

// lmax is 0..12. The comparison launch uses (components * count + 255) / 256
// blocks stored in an int. Counts whose buffers would overflow are rejected.
inline bool ylm_launch_ok(size_t count, int lmax, long long iterations)
{
    if (count == 0 || lmax < 0 || lmax > 12 || iterations <= 0) return false;
    const unsigned long long components =
        static_cast<unsigned long long>(lmax + 1) *
        static_cast<unsigned long long>(lmax + 1);
    const unsigned long long per_vector = components > 3 ? components : 3;
    unsigned long long max_count =
        static_cast<unsigned long long>(INT_MAX) * 256 / components;
    const unsigned long long max_bytes_count =
        static_cast<unsigned long long>(SIZE_MAX) / (per_vector * sizeof(double));
    if (max_bytes_count < max_count) max_count = max_bytes_count;
    return static_cast<unsigned long long>(count) <= max_count;
}

// Largest per-axis cell count whose edge is still at least the cutoff.
// A pair inside the cutoff then lies in the home cell or one of the 26
// adjacent cells. Truncating 1/cutoff is not enough: the value is not always
// integral in double, so the result is adjusted until n * cutoff <= 1.
inline int neighbor_cells_per_axis(double cutoff)
{
    const double inverse = 1.0 / cutoff;
    int cells = inverse >= static_cast<double>(INT_MAX)
                    ? INT_MAX
                    : static_cast<int>(inverse);
    if (cells < 1) cells = 1;
    if (cells < INT_MAX && static_cast<double>(cells + 1) * cutoff <= 1.0)
        ++cells;
    while (cells > 1 && static_cast<double>(cells) * cutoff > 1.0)
        --cells;
    return cells;
}

// One optimized block stages this many home particles from the same cell.
inline constexpr int kNeighborBlock = 64;
// Below this occupancy, staging a cell costs more than it reuses.
inline constexpr int kNeighborSharedMin = 16;

// Particles are points in the unit cube, so the cutoff is at most 1.
// The cell grid is indexed by a signed 32-bit count.
inline bool neighborlist_launch_ok(int particle_count, double cutoff,
                                   long long iterations)
{
    if (particle_count <= 0 || !(cutoff > 0.0) || cutoff > 1.0 || iterations <= 0)
        return false;
    const int cells = neighbor_cells_per_axis(cutoff);
    const unsigned long long grid =
        static_cast<unsigned long long>(cells) * cells * cells;
    return grid <= static_cast<unsigned long long>(INT_MAX);
}

int run_ylm(size_t count, int lmax, long long iterations);
int run_neighborlist(int particle_count, double cutoff, long long iterations);
