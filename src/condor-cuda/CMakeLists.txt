add_hecbench_benchmark(
    NAME condor
    MODEL cuda
    SOURCES main.cu
    CATEGORIES simulation
    COMPILE_OPTIONS -std=c++17 -Xcompiler=-fopenmp
    LINK_LIBRARIES -fopenmp
    TEST_ARGS --nx 8 --ny 8 --nz 10 --nodes 65 --iters 1 --warmup 0
    TEST_REGEX "PASS"
    TEST_TIMEOUT 120
)
