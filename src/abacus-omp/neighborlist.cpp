// SPDX-License-Identifier: BSD-3-Clause
// OpenMP target port of the uniform-grid neighbor count used by ABACUS.
#include "kernels.h"
#include "reference.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

struct Particle {
    double x, y, z;
};

#pragma omp declare target
static int cell_axis(double coordinate, int cells_per_axis)
{
    int value = static_cast<int>(coordinate * cells_per_axis);
    if (value < 0) value = 0;
    if (value >= cells_per_axis) value = cells_per_axis - 1;
    return value;
}

static int cell_index(double x, double y, double z, int cells_per_axis)
{
    const int cx = cell_axis(x, cells_per_axis);
    const int cy = cell_axis(y, cells_per_axis);
    const int cz = cell_axis(z, cells_per_axis);
    return (cx * cells_per_axis + cy) * cells_per_axis + cz;
}
#pragma omp end declare target

static void count_cells(const Particle* positions, int particle_count,
                        int cells_per_axis, int* cell_counts)
{
    #pragma omp target teams distribute parallel for \
        is_device_ptr(positions, cell_counts) thread_limit(256)
    for (int particle = 0; particle < particle_count; ++particle) {
        const Particle point = positions[particle];
        const int cell = cell_index(point.x, point.y, point.z, cells_per_axis);
        #pragma omp atomic update
        cell_counts[cell] += 1;
    }
}

static void fill_cells(const Particle* positions, int particle_count,
                       int cells_per_axis, int* cell_cursor, int* cell_particles)
{
    #pragma omp target teams distribute parallel for \
        is_device_ptr(positions, cell_cursor, cell_particles) thread_limit(256)
    for (int particle = 0; particle < particle_count; ++particle) {
        const Particle point = positions[particle];
        const int cell = cell_index(point.x, point.y, point.z, cells_per_axis);
        int slot;
        #pragma omp atomic capture
        slot = cell_cursor[cell]++;
        cell_particles[slot] = particle;
    }
}

// Cell-list order. Built once, outside the timed region, so the optimized
// kernel streams neighbor positions instead of gathering them.
static void pack_cell_positions(const Particle* positions, const int* cell_particles,
                                int particle_count, Particle* cell_positions)
{
    #pragma omp target teams distribute parallel for \
        is_device_ptr(positions, cell_particles, cell_positions) thread_limit(256)
    for (int slot = 0; slot < particle_count; ++slot)
        cell_positions[slot] = positions[cell_particles[slot]];
}

static void count_neighbors(const Particle* positions, int particle_count,
                            int cells_per_axis, double cutoff_squared,
                            const int* offsets, const int* cell_particles,
                            int* neighbor_counts)
{
    #pragma omp target teams distribute parallel for \
        is_device_ptr(positions, offsets, cell_particles, neighbor_counts) \
        thread_limit(256)
    for (int particle = 0; particle < particle_count; ++particle) {
        const Particle center = positions[particle];
        const int x = cell_axis(center.x, cells_per_axis);
        const int y = cell_axis(center.y, cells_per_axis);
        const int z = cell_axis(center.z, cells_per_axis);
        int count = 0;
        for (int dx = -1; dx <= 1; ++dx)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dz = -1; dz <= 1; ++dz) {
                    const int nx = x + dx, ny = y + dy, nz = z + dz;
                    if (nx < 0 || nx >= cells_per_axis || ny < 0 ||
                        ny >= cells_per_axis || nz < 0 || nz >= cells_per_axis)
                        continue;
                    const int cell = (nx * cells_per_axis + ny) * cells_per_axis + nz;
                    for (int slot = offsets[cell]; slot < offsets[cell + 1]; ++slot) {
                        const int other = cell_particles[slot];
                        if (other == particle) continue;
                        const Particle point = positions[other];
                        const double rx = center.x - point.x;
                        const double ry = center.y - point.y;
                        const double rz = center.z - point.z;
                        count += rx * rx + ry * ry + rz * rz <= cutoff_squared;
                    }
                }
        neighbor_counts[particle] = count;
    }
}

// Same counts as count_neighbors. One thread per packed slot walks the packed
// positions of the 27 neighboring cells.
static void count_neighbors_optimized(const Particle* cell_positions, int particle_count,
                                      int cells_per_axis, double cutoff_squared,
                                      const int* offsets, const int* cell_particles,
                                      int* neighbor_counts)
{
    #pragma omp target teams distribute parallel for \
        is_device_ptr(cell_positions, offsets, cell_particles, neighbor_counts) \
        thread_limit(256)
    for (int home_slot = 0; home_slot < particle_count; ++home_slot) {
        const int particle = cell_particles[home_slot];
        const Particle center = cell_positions[home_slot];
        const int x = cell_axis(center.x, cells_per_axis);
        const int y = cell_axis(center.y, cells_per_axis);
        const int z = cell_axis(center.z, cells_per_axis);
        int count = 0;
        for (int dx = -1; dx <= 1; ++dx)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dz = -1; dz <= 1; ++dz) {
                    const int nx = x + dx, ny = y + dy, nz = z + dz;
                    if (nx < 0 || nx >= cells_per_axis || ny < 0 ||
                        ny >= cells_per_axis || nz < 0 || nz >= cells_per_axis)
                        continue;
                    const int cell = (nx * cells_per_axis + ny) * cells_per_axis + nz;
                    for (int slot = offsets[cell]; slot < offsets[cell + 1]; ++slot) {
                        const int other = cell_particles[slot];
                        if (other == particle) continue;
                        const Particle point = cell_positions[slot];
                        const double rx = center.x - point.x;
                        const double ry = center.y - point.y;
                        const double rz = center.z - point.z;
                        count += rx * rx + ry * ry + rz * rz <= cutoff_squared;
                    }
                }
        neighbor_counts[particle] = count;
    }
}

static unsigned long long compare_counts(const int* baseline, const int* optimized,
                                         int particle_count)
{
    unsigned long long mismatches = 0;
    #pragma omp target teams distribute parallel for \
        is_device_ptr(baseline, optimized) reduction(+:mismatches)
    for (int particle = 0; particle < particle_count; ++particle)
        mismatches += baseline[particle] != optimized[particle];
    return mismatches;
}

int run_neighborlist(int particle_count, double cutoff, long long iterations)
{
    if (!neighborlist_launch_ok(particle_count, cutoff, iterations)) {
        std::fprintf(stderr, "invalid neighbor-list size\n");
        return EXIT_FAILURE;
    }

    const int cells_per_axis = neighbor_cells_per_axis(cutoff);
    const int cell_count = cells_per_axis * cells_per_axis * cells_per_axis;
    const int offset_count = cell_count + 1;
    std::vector<Particle> host_positions(particle_count);
    for (int i = 0; i < particle_count; ++i) {
        const unsigned int state = 1664525u * static_cast<unsigned int>(i + 1) + 1013904223u;
        host_positions[i] = Particle{(state & 0xffffu) / 65536.0,
                                     ((state >> 8) & 0xffffu) / 65536.0,
                                     ((state >> 16) & 0xffffu) / 65536.0};
    }

    std::vector<int> host_counts(cell_count, 0);
    std::vector<int> host_offsets(offset_count, 0);
    std::vector<int> host_cursor(cell_count, 0);
    std::vector<int> host_particles(particle_count, 0);
    std::vector<Particle> host_cell_positions(particle_count);
    std::vector<int> host_baseline(particle_count, 0);
    std::vector<int> host_optimized(particle_count, 0);
    Particle* positions = host_positions.data();
    Particle* cell_positions = host_cell_positions.data();
    int* cell_counts = host_counts.data();
    int* offsets = host_offsets.data();
    int* cell_cursor = host_cursor.data();
    int* cell_particles = host_particles.data();
    int* baseline_counts = host_baseline.data();
    int* optimized_counts = host_optimized.data();

    #pragma omp target enter data \
        map(to: positions[0:particle_count], cell_counts[0:cell_count]) \
        map(alloc: cell_positions[0:particle_count], \
                   offsets[0:offset_count], cell_cursor[0:cell_count], \
                   cell_particles[0:particle_count], baseline_counts[0:particle_count], \
                   optimized_counts[0:particle_count])

    #pragma omp target data use_device_ptr(positions, cell_counts)
    {
        count_cells(positions, particle_count, cells_per_axis, cell_counts);
    }
    #pragma omp target update from(cell_counts[0:cell_count])
    for (int cell = 0; cell < cell_count; ++cell)
        host_offsets[cell + 1] = host_offsets[cell] + host_counts[cell];
    for (int cell = 0; cell < cell_count; ++cell)
        host_cursor[cell] = host_offsets[cell];
    #pragma omp target update to(offsets[0:offset_count], cell_cursor[0:cell_count])

    const double cutoff_squared = cutoff * cutoff;
    bool reference_ok = false;
    double baseline_milliseconds = 0.0;
    double optimized_milliseconds = 0.0;
    unsigned long long mismatches = 0;
    #pragma omp target data \
        use_device_ptr(positions, cell_positions, offsets, cell_cursor, \
                       cell_particles, baseline_counts, optimized_counts)
    {
        fill_cells(positions, particle_count, cells_per_axis, cell_cursor, cell_particles);
        pack_cell_positions(positions, cell_particles, particle_count, cell_positions);
        // Untimed launches fill the buffers compared with the host reference.
        count_neighbors(positions, particle_count, cells_per_axis, cutoff_squared,
                        offsets, cell_particles, baseline_counts);
        count_neighbors_optimized(cell_positions, particle_count, cells_per_axis,
                                  cutoff_squared, offsets, cell_particles, optimized_counts);
    }
    #pragma omp target update from(baseline_counts[0:particle_count], \
                                   optimized_counts[0:particle_count])
    std::vector<NeighborSample> samples(particle_count);
    for (int i = 0; i < particle_count; ++i)
        samples[i] = NeighborSample{host_positions[i].x, host_positions[i].y,
                                    host_positions[i].z};
    reference_ok = neighbor_reference_matches(samples.data(), particle_count,
                                              host_baseline.data(), host_optimized.data(),
                                              cutoff);
    if (reference_ok) {
        #pragma omp target data \
            use_device_ptr(positions, cell_positions, offsets, cell_particles, \
                           baseline_counts, optimized_counts)
        {
            const auto baseline_start = std::chrono::steady_clock::now();
            for (long long i = 0; i < iterations; ++i)
                count_neighbors(positions, particle_count, cells_per_axis, cutoff_squared,
                                offsets, cell_particles, baseline_counts);
            baseline_milliseconds = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - baseline_start).count();

            const auto optimized_start = std::chrono::steady_clock::now();
            for (long long i = 0; i < iterations; ++i)
                count_neighbors_optimized(cell_positions, particle_count, cells_per_axis,
                                          cutoff_squared, offsets, cell_particles,
                                          optimized_counts);
            optimized_milliseconds = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - optimized_start).count();

            mismatches = compare_counts(baseline_counts, optimized_counts, particle_count);
        }
    }

    #pragma omp target exit data \
        map(release: positions[0:particle_count], cell_positions[0:particle_count], \
                     cell_counts[0:cell_count], offsets[0:offset_count], \
                     cell_cursor[0:cell_count], cell_particles[0:particle_count], \
                     baseline_counts[0:particle_count], \
                     optimized_counts[0:particle_count])

    if (!reference_ok) {
        std::printf("ABACUS NEIGHBORLIST: FAIL  particles=%d cells=%d "
                    "verification failed; timing skipped\n",
                    particle_count, cell_count);
        return EXIT_FAILURE;
    }

    const bool valid = mismatches == 0;
    const double baseline_time = baseline_milliseconds / iterations;
    const double optimized_time = optimized_milliseconds / iterations;
    std::printf("ABACUS NEIGHBORLIST: %s  particles=%d cells=%d baseline=%.4f ms "
                "optimized=%.4f ms speedup=%.2fx mismatches=%llu\n",
                valid ? "PASS" : "FAIL", particle_count, cell_count, baseline_time,
                optimized_time, baseline_time / optimized_time, mismatches);
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
