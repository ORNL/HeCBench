// SPDX-License-Identifier: BSD-3-Clause

#include <cerrno>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "kernels.h"

struct YlmCase {
    size_t vectors;
    int lmax;
};

// ABACUS develop commit 2d1da705. One launch is one k-point, or the density
// grid for ultrasoft augmentation.
// Si16: examples/34_gpu/01_pw_Si16, l_max=1, Gamma of the 5x5x5 mesh.
// Pt:   examples/28_efield/01_Pt_slab, l_max=3, Gamma.
// Fe:   examples/16_uspp/01_pw_Fe, Q(G) uses lmax=2*lmaxkb on the density grid.
static const YlmCase kDefaultYlm[] = {
    {16625, 1},
    {28191, 3},
    {5473, 4},
};
static constexpr int kDefaultParticles = 1 << 18;
static constexpr double kDefaultCutoff = 0.04;
static constexpr long long kDefaultIterations = 100;

static void print_usage(const char* program)
{
    std::fprintf(stderr,
                 "usage: %s [iterations]\n"
                 "       %s <vector-count> <0 <= maximum-l <= 12> <particle-count> "
                 "<0 < cutoff <= 1, cell grid fits in 32 bits> <iterations>\n"
                 "With no size arguments, runs spherical harmonics for the Si16, "
                 "Pt, and Fe example sizes, then the neighbor list "
                 "(%d particles, cutoff %.2f).\n",
                 program, program, kDefaultParticles, kDefaultCutoff);
}

// Accepts an optional leading minus and digits. Rejects junk and overflow.
static bool parse_integer(const char* text, long long& value)
{
    if (text == nullptr || text[0] == '\0') return false;
    if (text[0] != '-' && (text[0] < '0' || text[0] > '9')) return false;
    errno = 0;
    char* end = nullptr;
    const long long parsed = std::strtoll(text, &end, 10);
    if (end == text || *end != '\0' || errno == ERANGE) return false;
    value = parsed;
    return true;
}

static bool parse_cutoff(const char* text, double& value)
{
    if (text == nullptr || text[0] == '\0') return false;
    errno = 0;
    char* end = nullptr;
    const double parsed = std::strtod(text, &end);
    if (end == text || *end != '\0' || errno == ERANGE) return false;
    value = parsed;
    return true;
}

// Runs every spherical-harmonic size, then the neighbor list. Prints one
// summary after the per-kernel lines.
static int run_suite(const YlmCase* cases, int case_count, int particles,
                     double cutoff, long long iterations)
{
    int status = EXIT_SUCCESS;
    for (int i = 0; i < case_count; ++i) {
        if (run_ylm(cases[i].vectors, cases[i].lmax, iterations) != EXIT_SUCCESS)
            status = EXIT_FAILURE;
    }
    if (run_neighborlist(particles, cutoff, iterations) != EXIT_SUCCESS)
        status = EXIT_FAILURE;
    std::printf("ABACUS: %s\n", status == EXIT_SUCCESS ? "PASS" : "FAIL");
    return status;
}

int main(int argc, char** argv)
{
    const char* program = argc > 0 ? argv[0] : "abacus";
    if (argc == 1) {
        return run_suite(kDefaultYlm, 3, kDefaultParticles, kDefaultCutoff,
                         kDefaultIterations);
    }
    if (argc == 2) {
        long long iterations = 0;
        if (!parse_integer(argv[1], iterations) || iterations <= 0) {
            print_usage(program);
            return EXIT_FAILURE;
        }
        return run_suite(kDefaultYlm, 3, kDefaultParticles, kDefaultCutoff,
                         iterations);
    }
    if (argc == 6) {
        long long vectors = 0;
        long long lmax = 0;
        long long particles = 0;
        long long iterations = 0;
        double cutoff = 0.0;
        if (!parse_integer(argv[1], vectors) || !parse_integer(argv[2], lmax) ||
            !parse_integer(argv[3], particles) || !parse_cutoff(argv[4], cutoff) ||
            !parse_integer(argv[5], iterations) || vectors <= 0 ||
            static_cast<unsigned long long>(vectors) >
                static_cast<unsigned long long>(SIZE_MAX) ||
            lmax > 12 || particles <= 0 || particles > INT_MAX ||
            !ylm_launch_ok(static_cast<size_t>(vectors), static_cast<int>(lmax),
                           iterations) ||
            !neighborlist_launch_ok(static_cast<int>(particles), cutoff, iterations)) {
            print_usage(program);
            return EXIT_FAILURE;
        }
        const YlmCase one{static_cast<size_t>(vectors), static_cast<int>(lmax)};
        return run_suite(&one, 1, static_cast<int>(particles), cutoff, iterations);
    }
    print_usage(program);
    return EXIT_FAILURE;
}
