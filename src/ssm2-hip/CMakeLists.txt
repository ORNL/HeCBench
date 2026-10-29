add_hecbench_benchmark(
    NAME ssm2
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
    COMPILE_OPTIONS -fopenmp
    LINK_LIBRARIES -fopenmp
    TEST_REGEX "PASS"
    TEST_ARGS 1
)
