# kda-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME kda
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
    COMPILE_OPTIONS -Xcompiler=-fopenmp
    LINK_LIBRARIES -Xcompiler=-fopenmp
    TEST_REGEX "PASS"
    TEST_ARGS 1 16 2
)
