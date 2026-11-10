// SPDX-License-Identifier: BSD-3-Clause
// Independent host evaluation of the real spherical harmonics compared with
// the baseline and optimized kernels.

#include "reference.h"

#include "kernels.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr double pi = 3.141592653589793238462643383279502884;

// n! as a double. n is at most 24 because lmax is at most 12.
double series_factorial(int n)
{
    double result = 1.0;
    for (int value = 2; value <= n; ++value) result *= value;
    return result;
}

// Azimuth of the projection onto the xy plane. A radius whose |x| is at most
// 1e-9, including the origin, uses ±pi/2. Every other point uses atan2.
// sin(m*phi) and cos(m*phi) then agree with the kernels, whose third-quadrant
// angle differs from atan2 by 2*pi.
double azimuth(double x, double y)
{
    if (std::fabs(x) <= 1.0e-9) return y >= 0.0 ? pi / 2.0 : -pi / 2.0;
    return std::atan2(y, x);
}

}  // namespace

// Normalized, Condon-Shortley-phased coefficients of d^m P_l(x) / dx^m.
YlmReference::YlmReference(int lmax) : lmax_(lmax)
{
    for (int l = 0; l <= lmax_; ++l) {
        const double normalization = std::sqrt((2.0 * l + 1.0) / (4.0 * pi));
        for (int m = 0; m <= l; ++m) {
            const double scale = m == 0 ? normalization
                : normalization * std::sqrt(2.0) *
                  std::sqrt(series_factorial(l - m) / series_factorial(l + m));
            const double phase = (m & 1) ? -1.0 : 1.0;
            const int parity = (l - m) & 1;
            const int terms = (l - m - parity) / 2 + 1;
            const size_t first = coefficients_.size();
            coefficients_.resize(first + terms);
            for (int k = 0; 2 * k <= l - m; ++k) {
                const int j = (l - m - 2 * k - parity) / 2;
                coefficients_[first + j] =
                    phase * scale * ((k & 1) ? -1.0 : 1.0) *
                    series_factorial(2 * l - 2 * k) /
                    (std::ldexp(1.0, l) * series_factorial(k) *
                     series_factorial(l - k) *
                     series_factorial(l - m - 2 * k));
            }
            const int trig_kinds = m == 0 ? 1 : 2;
            for (int kind = 0; kind < trig_kinds; ++kind) {
                order_.push_back(m);
                trig_.push_back(m == 0 ? 0 : kind + 1);
                parity_.push_back(parity);
                first_.push_back(first);
                terms_.push_back(terms);
            }
        }
    }
}

// Component c equals x^parity * sum_j coefficient_j * (x^2)^j * sin(theta)^m,
// times cos(m*phi) or sin(m*phi). Here x is cos(theta).
void YlmReference::evaluate(double x, double y, double z,
                            std::vector<double>& harmonics) const
{
    const double magnitude = std::sqrt(x * x + y * y + z * z);
    const double cos_theta = magnitude < 1.0e-9 ? 0.0 : z / magnitude;
    const double sin_theta =
        std::sqrt(std::fmax(0.0, 1.0 - cos_theta * cos_theta));
    const double phi = azimuth(x, y);
    std::vector<double> sin_power(lmax_ + 1);
    std::vector<double> cos_m_phi(lmax_ + 1);
    std::vector<double> sin_m_phi(lmax_ + 1);
    for (int m = 0; m <= lmax_; ++m) {
        sin_power[m] = m == 0 ? 1.0 : sin_power[m - 1] * sin_theta;
        cos_m_phi[m] = std::cos(m * phi);
        sin_m_phi[m] = std::sin(m * phi);
    }
    const double t = cos_theta * cos_theta;
    harmonics.resize(order_.size());
    for (size_t c = 0; c < order_.size(); ++c) {
        double sum = 0.0;
        for (int j = terms_[c] - 1; j >= 0; --j)
            sum = sum * t + coefficients_[first_[c] + j];
        if (parity_[c]) sum *= cos_theta;
        const int m = order_[c];
        double value = sum * sin_power[m];
        if (trig_[c] == 1) value *= cos_m_phi[m];
        if (trig_[c] == 2) value *= sin_m_phi[m];
        harmonics[c] = value;
    }
}

// Closed form on +z: Y_l^0 = sqrt((2l+1)/4pi) and every m != 0 term is zero.
bool YlmReference::positive_z_pole(const std::vector<double>& column) const
{
    size_t component = 0;
    for (int l = 0; l <= lmax_; ++l) {
        const double expected = std::sqrt((2.0 * l + 1.0) / (4.0 * pi));
        const double tolerance = 1.0e-11 + 1.0e-10 * expected;
        if (!std::isfinite(column[component]) ||
            std::fabs(column[component] - expected) > tolerance)
            return false;
        ++component;
        for (int m = 1; m <= l; ++m) {
            if (!std::isfinite(column[component]) ||
                !std::isfinite(column[component + 1]) ||
                std::fabs(column[component]) > 1.0e-11 ||
                std::fabs(column[component + 1]) > 1.0e-11)
                return false;
            component += 2;
        }
    }
    return true;
}

// True when both results match. Vector 0 must also match +z, including the
// 1e-12 analytic check of Y_0^0 and Y_1^0.
bool YlmReference::matches_vector(size_t vector, double x, double y, double z,
                                  const std::vector<double>& baseline,
                                  const std::vector<double>& optimized) const
{
    std::vector<double> reference;
    evaluate(x, y, z, reference);
    for (size_t component = 0; component < reference.size(); ++component) {
        const double tolerance =
            1.0e-11 + 1.0e-10 * std::fabs(reference[component]);
        const bool baseline_ok = std::isfinite(baseline[component]) &&
            std::fabs(baseline[component] - reference[component]) <= tolerance;
        const bool optimized_ok = std::isfinite(optimized[component]) &&
            std::fabs(optimized[component] - reference[component]) <= tolerance;
        if (!baseline_ok || !optimized_ok) {
            std::fprintf(stderr,
                "reference mismatch vector=%zu component=%zu baseline=%.16g "
                "optimized=%.16g reference=%.16g\n",
                vector, component, baseline[component], optimized[component],
                reference[component]);
            return false;
        }
    }
    if (vector != 0) return true;
    if (!positive_z_pole(baseline) || !positive_z_pole(optimized)) {
        std::fprintf(stderr, "positive-z pole disagrees with the closed form\n");
        return false;
    }
    const double y00 = 1.0 / std::sqrt(4.0 * pi);
    const double y10 = std::sqrt(3.0 / (4.0 * pi));
    const auto within_analytic = [](double value, double expected) {
        return std::isfinite(value) && std::fabs(value - expected) < 1.0e-12;
    };
    if (!within_analytic(baseline[0], y00) ||
        !within_analytic(optimized[0], y00) ||
        (lmax_ > 0 && (!within_analytic(baseline[1], y10) ||
                       !within_analytic(optimized[1], y10)))) {
        std::fprintf(stderr,
            "positive-z Y00/Y10 outside 1e-12 of the closed form\n");
        return false;
    }
    return true;
}

bool YlmReference::matches_range(const std::vector<double>& host_vectors,
                                 size_t first, size_t chunk,
                                 const double* baseline, const double* optimized,
                                 size_t pitch) const
{
    const size_t components = order_.size();
    std::vector<double> baseline_column(components);
    std::vector<double> optimized_column(components);
    for (size_t j = 0; j < chunk; ++j) {
        for (size_t c = 0; c < components; ++c) {
            baseline_column[c] = baseline[c * pitch + j];
            optimized_column[c] = optimized[c * pitch + j];
        }
        const size_t vector = first + j;
        if (!matches_vector(vector, host_vectors[3 * vector],
                            host_vectors[3 * vector + 1],
                            host_vectors[3 * vector + 2], baseline_column,
                            optimized_column))
            return false;
    }
    return true;
}

namespace {

int sample_axis(double coordinate, int cells)
{
    int value = static_cast<int>(coordinate * static_cast<double>(cells));
    if (value < 0) value = 0;
    if (value >= cells) value = cells - 1;
    return value;
}

int sample_cell(const NeighborSample& point, int cells)
{
    const int x = sample_axis(point.x, cells);
    const int y = sample_axis(point.y, cells);
    const int z = sample_axis(point.z, cells);
    return (x * cells + y) * cells + z;
}

}

bool neighbor_reference_matches(const NeighborSample* positions, int particle_count,
                                const int* baseline, const int* optimized,
                                double cutoff)
{
    if (positions == nullptr || baseline == nullptr || optimized == nullptr ||
        particle_count <= 0 || !(cutoff > 0.0) || cutoff > 1.0)
        return false;
    const int cells = neighbor_cells_per_axis(cutoff);
    const unsigned long long grid =
        static_cast<unsigned long long>(cells) * cells * cells;
    if (grid > static_cast<unsigned long long>(INT_MAX)) return false;
    const int cell_count = static_cast<int>(grid);
    const double cutoff_squared = cutoff * cutoff;
    // A cell d steps away can hold a point inside the cutoff when the
    // separation is just over (d - 1) edges. The radius is the smallest
    // integer at least cutoff * cells, and at least 1, so a short edge
    // widens this walk past the device's 27-cell stencil.
    const double span = cutoff * static_cast<double>(cells);
    int reach = static_cast<int>(span);
    if (static_cast<double>(reach) < span)
        ++reach;
    if (reach < 1) reach = 1;

    std::vector<int> cell_counts(cell_count, 0);
    std::vector<int> cell_of(particle_count);
    for (int particle = 0; particle < particle_count; ++particle) {
        const int cell = sample_cell(positions[particle], cells);
        cell_of[particle] = cell;
        ++cell_counts[cell];
    }
    std::vector<int> offsets(cell_count + 1, 0);
    for (int cell = 0; cell < cell_count; ++cell)
        offsets[cell + 1] = offsets[cell] + cell_counts[cell];
    std::vector<int> cursor(offsets);
    std::vector<int> slots(particle_count);
    for (int particle = 0; particle < particle_count; ++particle)
        slots[cursor[cell_of[particle]]++] = particle;

    const int plane = cells * cells;
    for (int particle = 0; particle < particle_count; ++particle) {
        const int cell = cell_of[particle];
        const int x = cell / plane;
        const int remainder = cell - x * plane;
        const int y = remainder / cells;
        const int z = remainder - y * cells;
        const double px = positions[particle].x;
        const double py = positions[particle].y;
        const double pz = positions[particle].z;
        int expected = 0;
        for (int dx = -reach; dx <= reach; ++dx) {
            const int nx = x + dx;
            if (nx < 0 || nx >= cells) continue;
            for (int dy = -reach; dy <= reach; ++dy) {
                const int ny = y + dy;
                if (ny < 0 || ny >= cells) continue;
                for (int dz = -reach; dz <= reach; ++dz) {
                    const int nz = z + dz;
                    if (nz < 0 || nz >= cells) continue;
                    const int neighbor = (nx * cells + ny) * cells + nz;
                    for (int slot = offsets[neighbor]; slot < offsets[neighbor + 1]; ++slot) {
                        const int other = slots[slot];
                        if (other == particle) continue;
                        const double rx = px - positions[other].x;
                        const double ry = py - positions[other].y;
                        const double rz = pz - positions[other].z;
                        expected += rx * rx + ry * ry + rz * rz <= cutoff_squared;
                    }
                }
            }
        }
        if (baseline[particle] != expected || optimized[particle] != expected)
            return false;
    }

    const int sample = particle_count < 512 ? particle_count : 512;
    for (int particle = 0; particle < sample; ++particle) {
        int brute = 0;
        const double px = positions[particle].x;
        const double py = positions[particle].y;
        const double pz = positions[particle].z;
        for (int other = 0; other < particle_count; ++other) {
            if (other == particle) continue;
            const double rx = px - positions[other].x;
            const double ry = py - positions[other].y;
            const double rz = pz - positions[other].z;
            brute += rx * rx + ry * ry + rz * rz <= cutoff_squared;
        }
        if (brute != baseline[particle] || brute != optimized[particle])
            return false;
    }
    return true;
}
