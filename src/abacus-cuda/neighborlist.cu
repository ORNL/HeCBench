// SPDX-License-Identifier: BSD-3-Clause

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include "kernels.h"
#include "reference.h"

#define CUDA_CHECK(call) do {                                                    \
    cudaError_t error__ = (call);                                                \
    if (error__ != cudaSuccess) {                                                \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                     cudaGetErrorString(error__));                               \
        std::exit(EXIT_FAILURE);                                                 \
    }                                                                            \
} while (0)

__device__ int cell_index(const double3 point, int cells_per_axis)
{
    const int x = min(cells_per_axis - 1, max(0, static_cast<int>(point.x * cells_per_axis)));
    const int y = min(cells_per_axis - 1, max(0, static_cast<int>(point.y * cells_per_axis)));
    const int z = min(cells_per_axis - 1, max(0, static_cast<int>(point.z * cells_per_axis)));
    return (x * cells_per_axis + y) * cells_per_axis + z;
}

__global__ void count_cells(const double3* positions, int particle_count,
                            int cells_per_axis, int* cell_counts)
{
    const int particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle < particle_count)
        atomicAdd(&cell_counts[cell_index(positions[particle], cells_per_axis)], 1);
}

__global__ void fill_cells(const double3* positions, int particle_count,
                           int cells_per_axis, int* cell_cursor, int* cell_particles)
{
    const int particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle < particle_count) {
        const int cell = cell_index(positions[particle], cells_per_axis);
        cell_particles[atomicAdd(&cell_cursor[cell], 1)] = particle;
    }
}

// Cell-list order. Built once, outside the timed region, so the optimized
// kernels stream neighbor positions instead of gathering them.
__global__ void pack_cell_positions(const double3* __restrict__ positions,
                                    const int* __restrict__ cell_particles,
                                    int particle_count,
                                    double3* __restrict__ cell_positions)
{
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot < particle_count)
        cell_positions[slot] = positions[cell_particles[slot]];
}

__global__ void count_neighbors(const double3* positions, int particle_count,
                                int cells_per_axis, double cutoff_squared,
                                const int* offsets, const int* cell_particles,
                                int* neighbor_counts)
{
    const int particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= particle_count) return;

    const double3 center = positions[particle];
    const int x = min(cells_per_axis - 1, max(0, static_cast<int>(center.x * cells_per_axis)));
    const int y = min(cells_per_axis - 1, max(0, static_cast<int>(center.y * cells_per_axis)));
    const int z = min(cells_per_axis - 1, max(0, static_cast<int>(center.z * cells_per_axis)));
    int count = 0;
    for (int dx = -1; dx <= 1; ++dx)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dz = -1; dz <= 1; ++dz) {
                const int nx = x + dx, ny = y + dy, nz = z + dz;
                if (nx < 0 || nx >= cells_per_axis || ny < 0 || ny >= cells_per_axis ||
                    nz < 0 || nz >= cells_per_axis)
                    continue;
                const int cell = (nx * cells_per_axis + ny) * cells_per_axis + nz;
                for (int slot = offsets[cell]; slot < offsets[cell + 1]; ++slot) {
                    const int other = cell_particles[slot];
                    if (other == particle) continue;
                    const double3 point = positions[other];
                    const double rx = center.x - point.x;
                    const double ry = center.y - point.y;
                    const double rz = center.z - point.z;
                    count += rx * rx + ry * ry + rz * rz <= cutoff_squared;
                }
            }
    neighbor_counts[particle] = count;
}

// One block counts up to kNeighborBlock particles that share a cell.
struct NeighborTask {
    int slot_begin;
    int slot_end;
    int cell;
};

// Same counts as count_neighbors. Each block stages a neighbor cell once and
// reuses those positions for its home particles.
__global__ void count_neighbors_optimized(
    const NeighborTask* __restrict__ tasks,
    const double3* __restrict__ cell_positions, int cells_per_axis,
    double cutoff_squared, const int* __restrict__ offsets,
    const int* __restrict__ cell_particles, int* __restrict__ neighbor_counts)
{
    __shared__ double3 tile[kNeighborBlock];
    __shared__ int tile_ids[kNeighborBlock];

    const NeighborTask task = tasks[blockIdx.x];
    const int plane = cells_per_axis * cells_per_axis;
    const int x = task.cell / plane;
    const int remainder = task.cell - x * plane;
    const int y = remainder / cells_per_axis;
    const int z = remainder - y * cells_per_axis;
    const int home_slot = task.slot_begin + static_cast<int>(threadIdx.x);
    const bool active = home_slot < task.slot_end;
    int particle = 0;
    double3 center = make_double3(0.0, 0.0, 0.0);
    int count = 0;
    if (active) {
        particle = cell_particles[home_slot];
        center = cell_positions[home_slot];
    }

    for (int stencil = 0; stencil < 27; ++stencil) {
        const int dx = stencil / 9 - 1;
        const int dy = (stencil / 3) % 3 - 1;
        const int dz = stencil % 3 - 1;
        const int nx = x + dx;
        const int ny = y + dy;
        const int nz = z + dz;
        int begin = 0;
        int end = 0;
        if (nx >= 0 && nx < cells_per_axis && ny >= 0 && ny < cells_per_axis &&
            nz >= 0 && nz < cells_per_axis) {
            const int neighbor = (nx * cells_per_axis + ny) * cells_per_axis + nz;
            begin = offsets[neighbor];
            end = offsets[neighbor + 1];
        }
        for (int tile_base = begin; tile_base < end; tile_base += kNeighborBlock) {
            const int remaining = end - tile_base;
            const int tile_count = remaining < kNeighborBlock ? remaining : kNeighborBlock;
            if (threadIdx.x < tile_count) {
                tile_ids[threadIdx.x] = cell_particles[tile_base + threadIdx.x];
                tile[threadIdx.x] = cell_positions[tile_base + threadIdx.x];
            }
            __syncthreads();
            if (active) {
                for (int i = 0; i < tile_count; ++i) {
                    if (tile_ids[i] == particle) continue;
                    const double rx = center.x - tile[i].x;
                    const double ry = center.y - tile[i].y;
                    const double rz = center.z - tile[i].z;
                    count += rx * rx + ry * ry + rz * rz <= cutoff_squared;
                }
            }
            __syncthreads();
        }
    }
    if (active) neighbor_counts[particle] = count;
}

// Same counts, for cells too small to reuse a staged tile. One thread walks
// the packed positions of the 27 neighboring cells.
__global__ void count_neighbors_streamed(
    const int* __restrict__ home_slots, int home_count,
    const double3* __restrict__ cell_positions, int cells_per_axis,
    double cutoff_squared, const int* __restrict__ offsets,
    const int* __restrict__ cell_particles, int* __restrict__ neighbor_counts)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= home_count) return;

    const int home_slot = home_slots[index];
    const int particle = cell_particles[home_slot];
    const double3 center = cell_positions[home_slot];
    const int x = min(cells_per_axis - 1, max(0, static_cast<int>(center.x * cells_per_axis)));
    const int y = min(cells_per_axis - 1, max(0, static_cast<int>(center.y * cells_per_axis)));
    const int z = min(cells_per_axis - 1, max(0, static_cast<int>(center.z * cells_per_axis)));
    int count = 0;
    for (int dx = -1; dx <= 1; ++dx)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dz = -1; dz <= 1; ++dz) {
                const int nx = x + dx, ny = y + dy, nz = z + dz;
                if (nx < 0 || nx >= cells_per_axis || ny < 0 || ny >= cells_per_axis ||
                    nz < 0 || nz >= cells_per_axis)
                    continue;
                const int cell = (nx * cells_per_axis + ny) * cells_per_axis + nz;
                for (int slot = offsets[cell]; slot < offsets[cell + 1]; ++slot) {
                    const int other = cell_particles[slot];
                    if (other == particle) continue;
                    const double3 point = cell_positions[slot];
                    const double rx = center.x - point.x;
                    const double ry = center.y - point.y;
                    const double rz = center.z - point.z;
                    count += rx * rx + ry * ry + rz * rz <= cutoff_squared;
                }
            }
    neighbor_counts[particle] = count;
}

__global__ void compare_counts(const int* __restrict__ baseline,
                               const int* __restrict__ optimized, int particle_count,
                               unsigned int* __restrict__ mismatches)
{
    const int particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle < particle_count && baseline[particle] != optimized[particle])
        atomicAdd(mismatches, 1u);
}

int run_neighborlist(int particle_count, double cutoff, long long iterations)
{
    if (!neighborlist_launch_ok(particle_count, cutoff, iterations)) {
        std::fprintf(stderr, "invalid neighbor-list size\n");
        return EXIT_FAILURE;
    }

    const int cells_per_axis = neighbor_cells_per_axis(cutoff);
    const int cell_count = cells_per_axis * cells_per_axis * cells_per_axis;
    std::vector<double3> host_positions(particle_count);
    for (int i = 0; i < particle_count; ++i) {
        const unsigned int state = 1664525u * static_cast<unsigned int>(i + 1) + 1013904223u;
        host_positions[i] = make_double3((state & 0xffffu) / 65536.0,
                                         ((state >> 8) & 0xffffu) / 65536.0,
                                         ((state >> 16) & 0xffffu) / 65536.0);
    }

    double3 *positions = nullptr, *cell_positions = nullptr;
    int *cell_counts = nullptr, *cell_cursor = nullptr, *offsets = nullptr;
    int *cell_particles = nullptr, *baseline_counts = nullptr;
    int *optimized_counts = nullptr, *home_slots = nullptr;
    NeighborTask* tasks = nullptr;
    unsigned int* device_mismatches = nullptr;
    CUDA_CHECK(cudaMalloc(&positions, particle_count * sizeof(double3)));
    CUDA_CHECK(cudaMalloc(&cell_positions, particle_count * sizeof(double3)));
    CUDA_CHECK(cudaMalloc(&cell_counts, cell_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cell_cursor, cell_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&offsets, (cell_count + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cell_particles, particle_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&baseline_counts, particle_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&optimized_counts, particle_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&home_slots, particle_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&tasks, particle_count * sizeof(NeighborTask)));
    CUDA_CHECK(cudaMalloc(&device_mismatches, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(positions, host_positions.data(), particle_count * sizeof(double3),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(cell_counts, 0, cell_count * sizeof(int)));
    CUDA_CHECK(cudaMemset(baseline_counts, 0, particle_count * sizeof(int)));
    CUDA_CHECK(cudaMemset(optimized_counts, 0, particle_count * sizeof(int)));
    const int blocks = (particle_count + 255) / 256;
    count_cells<<<blocks, 256>>>(positions, particle_count, cells_per_axis, cell_counts);
    CUDA_CHECK(cudaGetLastError());

    std::vector<int> host_counts(cell_count), host_offsets(cell_count + 1);
    CUDA_CHECK(cudaMemcpy(host_counts.data(), cell_counts, cell_count * sizeof(int),
                          cudaMemcpyDeviceToHost));
    for (int cell = 0; cell < cell_count; ++cell)
        host_offsets[cell + 1] = host_offsets[cell] + host_counts[cell];
    std::vector<NeighborTask> host_tasks;
    std::vector<int> host_slots;
    host_tasks.reserve(static_cast<size_t>(particle_count));
    host_slots.reserve(static_cast<size_t>(particle_count));
    for (int cell = 0; cell < cell_count; ++cell) {
        const int begin = host_offsets[cell];
        const int end = host_offsets[cell + 1];
        if (end - begin >= kNeighborSharedMin) {
            for (int slot = begin; slot < end; slot += kNeighborBlock) {
                const int slot_end = slot + kNeighborBlock < end ? slot + kNeighborBlock : end;
                if (slot_end - slot >= kNeighborSharedMin)
                    host_tasks.push_back(NeighborTask{slot, slot_end, cell});
                else
                    for (int home = slot; home < slot_end; ++home) host_slots.push_back(home);
            }
        } else {
            for (int home = begin; home < end; ++home) host_slots.push_back(home);
        }
    }
    const int shared_count = static_cast<int>(host_tasks.size());
    const int streamed_count = static_cast<int>(host_slots.size());
    if (shared_count > 0)
        CUDA_CHECK(cudaMemcpy(tasks, host_tasks.data(), host_tasks.size() * sizeof(NeighborTask),
                              cudaMemcpyHostToDevice));
    if (streamed_count > 0)
        CUDA_CHECK(cudaMemcpy(home_slots, host_slots.data(), host_slots.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(offsets, host_offsets.data(), (cell_count + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cell_cursor, host_offsets.data(), cell_count * sizeof(int),
                          cudaMemcpyHostToDevice));
    fill_cells<<<blocks, 256>>>(positions, particle_count, cells_per_axis, cell_cursor,
                                cell_particles);
    CUDA_CHECK(cudaGetLastError());
    pack_cell_positions<<<blocks, 256>>>(positions, cell_particles, particle_count,
                                         cell_positions);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const double cutoff_squared = cutoff * cutoff;
    const int streamed_blocks = (streamed_count + 255) / 256;
    auto launch_optimized = [&]() {
        if (shared_count > 0)
            count_neighbors_optimized<<<shared_count, kNeighborBlock>>>(
                tasks, cell_positions, cells_per_axis, cutoff_squared, offsets,
                cell_particles, optimized_counts);
        if (streamed_count > 0)
            count_neighbors_streamed<<<streamed_blocks, 256>>>(
                home_slots, streamed_count, cell_positions, cells_per_axis, cutoff_squared,
                offsets, cell_particles, optimized_counts);
    };
    // Untimed launches fill the buffers compared with the host reference.
    count_neighbors<<<blocks, 256>>>(positions, particle_count, cells_per_axis,
                                     cutoff_squared, offsets, cell_particles,
                                     baseline_counts);
    CUDA_CHECK(cudaGetLastError());
    launch_optimized();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> host_baseline(particle_count), host_optimized(particle_count);
    CUDA_CHECK(cudaMemcpy(host_baseline.data(), baseline_counts, particle_count * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_optimized.data(), optimized_counts,
                          particle_count * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<NeighborSample> samples(particle_count);
    for (int i = 0; i < particle_count; ++i)
        samples[i] = NeighborSample{host_positions[i].x, host_positions[i].y,
                                    host_positions[i].z};
    const bool reference_ok = neighbor_reference_matches(
        samples.data(), particle_count, host_baseline.data(), host_optimized.data(), cutoff);
    if (!reference_ok) {
        std::printf("ABACUS NEIGHBORLIST: FAIL  particles=%d cells=%d "
                    "verification failed; timing skipped\n",
                    particle_count, cell_count);
        CUDA_CHECK(cudaFree(positions));
        CUDA_CHECK(cudaFree(cell_positions));
        CUDA_CHECK(cudaFree(cell_counts));
        CUDA_CHECK(cudaFree(cell_cursor));
        CUDA_CHECK(cudaFree(offsets));
        CUDA_CHECK(cudaFree(cell_particles));
        CUDA_CHECK(cudaFree(baseline_counts));
        CUDA_CHECK(cudaFree(optimized_counts));
        CUDA_CHECK(cudaFree(home_slots));
        CUDA_CHECK(cudaFree(tasks));
        CUDA_CHECK(cudaFree(device_mismatches));
        return EXIT_FAILURE;
    }

    const auto baseline_start = std::chrono::steady_clock::now();
    for (long long i = 0; i < iterations; ++i)
        count_neighbors<<<blocks, 256>>>(positions, particle_count, cells_per_axis,
                                         cutoff_squared, offsets, cell_particles,
                                         baseline_counts);
    CUDA_CHECK(cudaDeviceSynchronize());
    const double baseline_milliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - baseline_start).count();

    const auto optimized_start = std::chrono::steady_clock::now();
    for (long long i = 0; i < iterations; ++i) launch_optimized();
    CUDA_CHECK(cudaDeviceSynchronize());
    const double optimized_milliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - optimized_start).count();

    CUDA_CHECK(cudaMemset(device_mismatches, 0, sizeof(unsigned int)));
    compare_counts<<<blocks, 256>>>(baseline_counts, optimized_counts, particle_count,
                                    device_mismatches);
    CUDA_CHECK(cudaGetLastError());
    unsigned int mismatches = 0;
    CUDA_CHECK(cudaMemcpy(&mismatches, device_mismatches, sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));

    const bool valid = mismatches == 0;
    const double baseline_time = baseline_milliseconds / iterations;
    const double optimized_time = optimized_milliseconds / iterations;
    std::printf("ABACUS NEIGHBORLIST: %s  particles=%d cells=%d baseline=%.4f ms "
                "optimized=%.4f ms speedup=%.2fx mismatches=%u\n",
                valid ? "PASS" : "FAIL", particle_count, cell_count, baseline_time,
                optimized_time, baseline_time / optimized_time, mismatches);

    CUDA_CHECK(cudaFree(positions));
    CUDA_CHECK(cudaFree(cell_positions));
    CUDA_CHECK(cudaFree(cell_counts));
    CUDA_CHECK(cudaFree(cell_cursor));
    CUDA_CHECK(cudaFree(offsets));
    CUDA_CHECK(cudaFree(cell_particles));
    CUDA_CHECK(cudaFree(baseline_counts));
    CUDA_CHECK(cudaFree(optimized_counts));
    CUDA_CHECK(cudaFree(home_slots));
    CUDA_CHECK(cudaFree(tasks));
    CUDA_CHECK(cudaFree(device_mismatches));
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
