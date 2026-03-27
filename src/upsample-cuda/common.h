float* make_random_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    #pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
        arr[i] = rand() / (float)RAND_MAX * 2.f - 1.f; // range -1..1
    }
    return arr;
}

template<class T>
T ceil_div(T dividend, T divisor) {
    return (dividend + divisor-1) / divisor;
}

inline int max_int(int a, int b) {
    return a > b ? a : b;
}

inline int min_int(int a, int b) {
    return a < b ? a : b;
}
