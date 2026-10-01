#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sycl/sycl.hpp>

float* make_random_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = rand() / (float)RAND_MAX * 2.f - 1.f;
    }
    return arr;
}
