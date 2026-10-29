// Copyright (c) 2017 Lawrence Livermore National Security, LLC.
// SPDX-License-Identifier: BSD-2-Clause

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>
#include <sycl/sycl.hpp>
#include "../laghos-force-cuda/reference.hpp"


using namespace laghos_force;

template <int D1D, int Q1D, int L1D>
static inline void force_mult_3d(
    sycl::nd_item<1> item, int ne, const double *__restrict__ b,
    const double *__restrict__ bt, const double *__restrict__ gt,
    const double *__restrict__ stress, const double *__restrict__ energy,
    double *__restrict__ velocity, sycl::local_accessor<double, 1> B,
    sycl::local_accessor<double, 1> Bt, sycl::local_accessor<double, 1> Gt,
    sycl::local_accessor<double, 1> E, sycl::local_accessor<double, 1> EQ,
    sycl::local_accessor<double, 1> X, sycl::local_accessor<double, 1> XY) {
  constexpr int QQQ = Q1D * Q1D * Q1D;
  constexpr int DDD = D1D * D1D * D1D;
  constexpr int LLL = L1D * L1D * L1D;
  constexpr int kThreads = QQQ;
  const auto grp = item.get_group();
  const int e = item.get_group(0);
  const int t = item.get_local_id(0);
  if (e >= ne)
    return;

  if (t < Q1D * L1D)
    B[t] = b[t];
  if (t < D1D * Q1D) {
    Bt[t] = bt[t];
    Gt[t] = gt[t];
  }
  if (t < LLL)
    E[t] = energy[t + static_cast<std::size_t>(LLL) * e];
  sycl::group_barrier(grp);

  const int qx = t % Q1D;
  const int qy = (t / Q1D) % Q1D;
  const int qz = t / (Q1D * Q1D);
  double eq = 0.0;
#pragma unroll
  for (int lz = 0; lz < L1D; ++lz)
#pragma unroll
    for (int ly = 0; ly < L1D; ++ly)
#pragma unroll
      for (int lx = 0; lx < L1D; ++lx) {
        const int l = lx + L1D * (ly + L1D * lz);
        eq += B[qx * L1D + lx] * B[qy * L1D + ly] * B[qz * L1D + lz] * E[l];
      }
  EQ[t] = eq;
  sycl::group_barrier(grp);

  constexpr int nx = D1D * Q1D * Q1D;
  for (int n = t; n < DIM * DIM * nx; n += kThreads) {
    const int local = n % nx;
    const int pair = n / nx;
    const int hx = local % D1D;
    const int qy0 = (local / D1D) % Q1D;
    const int qz0 = local / (D1D * Q1D);
    const int i = pair % DIM;
    const int c = pair / DIM;
    double v = 0.0;
#pragma unroll
    for (int qx0 = 0; qx0 < Q1D; ++qx0) {
      const int q = qx0 + Q1D * (qy0 + Q1D * qz0);
      const double m = i == 0 ? Gt[hx * Q1D + qx0] : Bt[hx * Q1D + qx0];
      v += m * EQ[q] * stress[stress_index(q, e, i, c, ne, QQQ)];
    }
    X[n] = v;
  }
  sycl::group_barrier(grp);

  constexpr int nxy = D1D * D1D * Q1D;
  for (int n = t; n < DIM * DIM * nxy; n += kThreads) {
    const int local = n % nxy;
    const int pair = n / nxy;
    const int hx = local % D1D;
    const int hy = (local / D1D) % D1D;
    const int qz0 = local / (D1D * D1D);
    const int i = pair % DIM;
    double v = 0.0;
#pragma unroll
    for (int qy0 = 0; qy0 < Q1D; ++qy0) {
      const int xi = hx + D1D * (qy0 + Q1D * qz0) + nx * pair;
      const double m = i == 1 ? Gt[hy * Q1D + qy0] : Bt[hy * Q1D + qy0];
      v += X[xi] * m;
    }
    XY[n] = v;
  }
  sycl::group_barrier(grp);

  for (int n = t; n < DIM * DDD; n += kThreads) {
    const int d = n % DDD;
    const int c = n / DDD;
    const int hx = d % D1D;
    const int hy = (d / D1D) % D1D;
    const int hz = d / (D1D * D1D);
    double v = 0.0;
#pragma unroll
    for (int qz0 = 0; qz0 < Q1D; ++qz0) {
#pragma unroll
      for (int i = 0; i < DIM; ++i) {
        const int yi = hx + D1D * (hy + D1D * qz0) + nxy * (i + DIM * c);
        const double m = i == 2 ? Gt[hz * Q1D + qz0] : Bt[hz * Q1D + qz0];
        v += XY[yi] * m;
      }
    }
    velocity[output_index(d, c, e, DDD)] =
        (v > -kEpsSquared && v < kEpsSquared) ? 0.0 : v;
  }
}

static int integer_arg(const char *text, const char *name, long minimum) {
  char *end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (!text[0] || *end || value < minimum || value > 100000000L) {
    std::fprintf(stderr, "invalid %s: %s\n", name, text);
    std::exit(EXIT_FAILURE);
  }
  return static_cast<int>(value);
}

template <int D1D, int Q1D, int L1D>
int run_case(sycl::queue &q, const sycl::device &device, int elements,
             int iterations, int warmup) {
  constexpr int QQQ = Q1D * Q1D * Q1D;
  constexpr int DDD = D1D * D1D * D1D;
  constexpr int kThreads = QQQ;
  constexpr int nx = D1D * Q1D * Q1D;
  constexpr int nxy = D1D * D1D * Q1D;
  std::vector<double> b(Q1D * L1D), bt(D1D * Q1D), gt(D1D * Q1D);
  make_basis(D1D, Q1D, L1D, b.data(), bt.data(), gt.data());
  std::vector<double> energy, stress;
  initialize(elements, Q1D, L1D, energy, stress);
  std::vector<double> velocity(static_cast<std::size_t>(elements) * DIM * DDD);

  auto alloc = [&](std::size_t n) {
    double *p = sycl::malloc_device<double>(n, q);
    if (!p)
      throw std::bad_alloc();
    return p;
  };
  double *db = alloc(b.size());
  double *dbt = alloc(bt.size());
  double *dgt = alloc(gt.size());
  double *dstress = alloc(stress.size());
  double *denergy = alloc(energy.size());
  double *dvelocity = alloc(velocity.size());
  q.memcpy(db, b.data(), sizeof(double) * b.size());
  q.memcpy(dbt, bt.data(), sizeof(double) * bt.size());
  q.memcpy(dgt, gt.data(), sizeof(double) * gt.size());
  q.memcpy(dstress, stress.data(), sizeof(double) * stress.size());
  q.memcpy(denergy, energy.data(), sizeof(double) * energy.size()).wait();

  auto launch = [&]() {
    return q.submit([&](sycl::handler &h) {
      sycl::local_accessor<double, 1> B(Q1D * L1D, h);
      sycl::local_accessor<double, 1> Bt(D1D * Q1D, h);
      sycl::local_accessor<double, 1> Gt(D1D * Q1D, h);
      sycl::local_accessor<double, 1> E(L1D * L1D * L1D, h);
      sycl::local_accessor<double, 1> EQ(QQQ, h);
      sycl::local_accessor<double, 1> X(DIM * DIM * nx, h);
      sycl::local_accessor<double, 1> XY(DIM * DIM * nxy, h);
      h.parallel_for(
          sycl::nd_range<1>(static_cast<std::size_t>(elements) * kThreads,
                            kThreads),
          [=](sycl::nd_item<1> item) {
            force_mult_3d<D1D, Q1D, L1D>(item, elements, db, dbt, dgt, dstress,
                                         denergy, dvelocity, B, Bt, Gt, E, EQ,
                                         X, XY);
          });
    });
  };

  for (int i = 0; i < warmup; ++i)
    launch();
  q.wait_and_throw();

  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i)
    launch();
  q.wait();
  const auto stop = std::chrono::steady_clock::now();
  q.memcpy(velocity.data(), dvelocity, sizeof(double) * velocity.size())
      .wait();

  const double ms =
      std::chrono::duration<double, std::milli>(stop - start).count() /
      iterations;
  const int checked = std::min(elements, 8);
  const double error = max_relative_error(
      elements, checked, D1D, Q1D, L1D, b.data(), bt.data(), gt.data(),
      stress.data(), energy.data(), velocity.data());
  const double flops = flops_per_element(D1D, Q1D, L1D);
  const std::string name = device.get_info<sycl::info::device::name>();
  std::printf("device: %s\n", name.c_str());
  std::printf("points: %d  D1D: %d  Q1D: %d  L1D: %d\n", QQQ, D1D, Q1D, L1D);
  std::printf("elements: %d  iterations: %d  warmup: %d\n", elements,
              iterations, warmup);
  std::printf("kernel: %.6f ms  throughput: %.3f GFLOP/s\n", ms,
              flops * elements / (ms * 1.0e6));
  std::printf("max_relative_error: %.3e (%d elements checked)\n", error,
              checked);
  const bool pass = std::isfinite(error) && error < 2.0e-12;
  std::printf("laghos-force: %s\n", pass ? "PASS" : "FAIL");

  sycl::free(db, q);
  sycl::free(dbt, q);
  sycl::free(dgt, q);
  sycl::free(dstress, q);
  sycl::free(denergy, q);
  sycl::free(dvelocity, q);
  return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}

int main(int argc, char **argv) {
  int points = 64;
  int elements = 0;
  int iterations = 100;
  int warmup = 100;
  for (int i = 1; i < argc; ++i) {
    const bool has_value = i + 1 < argc;
    if (has_value && !std::strcmp(argv[i], "--points"))
      points = integer_arg(argv[++i], "sample points", 1);
    else if (has_value && !std::strcmp(argv[i], "--elements"))
      elements = integer_arg(argv[++i], "element count", 1);
    else if (has_value && !std::strcmp(argv[i], "--iters"))
      iterations = integer_arg(argv[++i], "iteration count", 1);
    else if (has_value && !std::strcmp(argv[i], "--warmup"))
      warmup = integer_arg(argv[++i], "warmup count", 0);
    else {
      std::fprintf(stderr,
                   "usage: %s [--points 8|64|216|512] [--elements N] "
                   "[--iters N] [--warmup N]\n",
                   argv[0]);
      return EXIT_FAILURE;
    }
  }
  if (!supported_points(points)) {
    std::fprintf(stderr, "sample points must be 8, 64, 216, or 512\n");
    return EXIT_FAILURE;
  }
  if (elements == 0)
    elements = default_elements(points);

  try {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    const sycl::device device = q.get_device();
    if (!device.has(sycl::aspect::fp64)) {
      std::fprintf(stderr, "selected device does not support fp64\n");
      return EXIT_FAILURE;
    }
    switch (points) {
    case 8:
      return run_case<2, 2, 1>(q, device, elements, iterations, warmup);
    case 64:
      return run_case<3, 4, 2>(q, device, elements, iterations, warmup);
    case 216:
      return run_case<4, 6, 3>(q, device, elements, iterations, warmup);
    case 512:
      return run_case<5, 8, 4>(q, device, elements, iterations, warmup);
    default:
      return EXIT_FAILURE;
    }
  } catch (const sycl::exception &ex) {
    std::fprintf(stderr, "SYCL error: %s\n", ex.what());
    return EXIT_FAILURE;
  }
}
