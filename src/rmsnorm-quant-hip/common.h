#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>

float* make_random_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = rand() / (float)RAND_MAX * 2.f - 1.f;
    }
    return arr;
}

void hip_check(hipError_t error, const char *file, int line) {
    if (error != hipSuccess) {
        printf("[HIP ERROR] at file %s:%d:\n%s\n", file, line,
               hipGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};
#define hipCheck(err) (hip_check(err, __FILE__, __LINE__))
