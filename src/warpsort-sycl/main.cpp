//
// Copyright 2004-present Facebook. All Rights Reserved.
//

#include <sycl/sycl.hpp>
#include <algorithm>
#include <random>
#include <unordered_set>
#include <vector>
#include <cstdlib>

namespace facebook { namespace cuda {

// test the warp-wide sort code
std::vector<float> sort(sycl::queue &q, const std::vector<float>& data, double &time);

// test the warp-wide sort with indices code
std::vector<std::pair<float, int> >
sortWithIndices(sycl::queue &q, const std::vector<float>& data, double &);

} } // namespace

// Add in +/- inf, +/- 0, denorms
void addSpecialFloats(std::vector<float>& vals) {
  // Add in +/- infinity, with duplicates
  vals.push_back(std::numeric_limits<float>::infinity());
  vals.push_back(std::numeric_limits<float>::infinity());
  vals.push_back(-std::numeric_limits<float>::infinity());
  vals.push_back(-std::numeric_limits<float>::infinity());

  // Add in +/- zero, with duplicates
  vals.push_back(0.0f);
  vals.push_back(0.0f);
  vals.push_back(-0.0f);
  vals.push_back(-0.0f);

  // Add in some denorm floats, with duplicates
  vals.push_back(std::numeric_limits<float>::denorm_min() * 4.0f);
  vals.push_back(std::numeric_limits<float>::denorm_min());
  vals.push_back(std::numeric_limits<float>::denorm_min());
  vals.push_back(-std::numeric_limits<float>::denorm_min());
  vals.push_back(-std::numeric_limits<float>::denorm_min());
  vals.push_back(-std::numeric_limits<float>::denorm_min() * 4.0f);
}

bool test_sort(sycl::queue &q, const int repeat) {
  std::vector<float> vals;
  addSpecialFloats(vals);

  std::vector<float> sorted = vals;
  std::sort(sorted.begin(), sorted.end(), std::greater<float>());

  double time = 0.0;

  bool ok = true;
  for (int i = 0; i < repeat; ++i) {
    std::shuffle(vals.begin(), vals.end(), std::random_device());
    auto out = facebook::cuda::sort(q, vals, time);

    if (sorted.size() != out.size()) {
      ok = false;
      goto DONE;
    }

    for (int j = 0; j < (int)out.size(); ++j) {
      if (sorted[j] != out[j]) {
        ok = false;
        goto DONE;
      }
    }
  }
  printf("Size = %3d | average kernel execution time: %f (us)\n",
         (int)sorted.size(), (time * 1e-3f) / repeat);
  DONE:
  return ok;
}

bool test_sortInRegisters(sycl::queue &q, const int repeat) {
  // Test sorting std::vectors of size 1 to 4 x warpSize, which is the
  // maximum in-register size we support
  bool ok = true;

  for (int size = 16; size <= 4 * 32; size = size * 2) {
    std::vector<float> vals;

    for (int i = 0; i < size; ++i) {
      vals.push_back((float) i + 1);
    }

    std::vector<float> sorted = vals;
    std::sort(sorted.begin(), sorted.end(), std::greater<float>());

    double time = 0.0;

    for (int i = 0; i < repeat; ++i) {
      std::shuffle(vals.begin(), vals.end(), std::random_device());
      auto out = facebook::cuda::sort(q, vals, time);

      if (sorted.size() != out.size()) {
        ok = false;
        goto DONE;
      }

      for (int j = 0; j < (int)out.size(); ++j) {
        if (sorted[j] != out[j]) {
          ok = false;
          goto DONE;
        }
      }
    }
    printf("Size = %3d | average kernel execution time: %f (us)\n",
           size, (time * 1e-3f) / repeat);
  }
  DONE:
  return ok;
}

bool test_sortIndicesInRegisters(sycl::queue &q, const int repeat) {
  // Test sorting std::vectors of size 1 to 4 x warpSize, which is the
  // maximum in-register size we support

  bool ok = true;
  for (int size = 16; size <= 4 * 32; size = size * 2) {
    std::vector<float> vals;

    for (int i = 0; i < size; ++i) {
      vals.push_back((float) i);
    }

    std::vector<float> sorted = vals;
    std::sort(sorted.begin(), sorted.end(), std::greater<float>());

    double time = 0.0;

    for (int i = 0; i < repeat; ++i) {
      std::shuffle(vals.begin(), vals.end(), std::random_device());
      auto out = facebook::cuda::sortWithIndices(q, vals, time);

      if (sorted.size() != out.size()) {
        ok = false;
        goto DONE;
      }

      for (int j = 0; j < (int)out.size(); ++j) {
        if (sorted[j] != out[j].first) {
          ok = false;
          goto DONE;
        }

        int idx = out[j].second;
        if (idx < 0 || idx >= (int)vals.size()  || out[j].first != vals[idx]) {
          ok = false;
          goto DONE;
        }
      }

      // Test for uniqueness of indices
      std::unordered_set<int> indices;
      for (const auto &p : out) {
        if (indices.count(p.second) == true) {
          ok = false;
          goto DONE;
        }
        indices.emplace(p.second);
      }
    }
    printf("Size = %3d | average kernel execution time: %f (us)\n",
           size, (time * 1e-3f) / repeat);
  }
  DONE:
  return ok;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  bool ok;
  ok = test_sort(q, repeat);
  printf("test_sort: %s\n\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);

  ok = test_sortInRegisters(q, repeat);
  printf("test_sortInRegisters: %s\n\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);

  ok = test_sortIndicesInRegisters(q, repeat);
  printf("test_sortIndicesInRegisters: %s\n\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);

  return 0;
}
