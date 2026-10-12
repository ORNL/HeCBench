add_hecbench_benchmark(
    NAME laghos-force
    MODEL cuda
    SOURCES main.cu
    CATEGORIES simulation
    TEST_ARGS --elements 128 --iters 2 --warmup 1
    TEST_REGEX "laghos-force: PASS"
)
