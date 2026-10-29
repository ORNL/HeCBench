add_hecbench_benchmark(
    NAME condor
    MODEL omp
    SOURCES main.cpp
    CATEGORIES simulation
    COMPILE_OPTIONS -std=c++17
    TEST_ARGS --nx 8 --ny 8 --nz 10 --nodes 65 --iters 1 --warmup 0
    TEST_REGEX "PASS"
    TEST_TIMEOUT 120
)
