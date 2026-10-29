add_hecbench_benchmark(
    NAME laghos-force
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES simulation
    TEST_ARGS --elements 128 --iters 2 --warmup 1
    TEST_REGEX "laghos-force: PASS"
)
