# shampoo-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME shampoo
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
    COMPILE_OPTIONS -fopenmp
    LINK_LIBRARIES hipblas -fopenmp
    TEST_ARGS 33 31 3 1
    TEST_REGEX "PASS"
)
