# hisparse-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME hisparse
    MODEL omp
    SOURCES main.cpp hisparse_kernels.cpp
    CATEGORIES ml
    TEST_ARGS 8 256 1
    TEST_REGEX "PASS"
)
