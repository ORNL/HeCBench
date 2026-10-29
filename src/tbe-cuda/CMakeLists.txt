add_hecbench_benchmark(
    NAME tbe
    MODEL cuda
    SOURCES main.cu
    CATEGORIES machine-learning
    TEST_ARGS 2 64 1024 64 8 1
)
