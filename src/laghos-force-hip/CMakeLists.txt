add_hecbench_benchmark(
    NAME laghos-force
    MODEL hip
    SOURCES main.cu
    CATEGORIES simulation
    TEST_ARGS --elements 128 --iters 2 --warmup 1
    TEST_REGEX "laghos-force: PASS"
)
