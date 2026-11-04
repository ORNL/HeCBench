add_hecbench_benchmark(
    NAME gis
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
    TEST_ARGS 4 1000 64 2048 2
)
