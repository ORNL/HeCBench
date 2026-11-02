# ssm2-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME ssm2
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
    COMPILE_OPTIONS -Xcompiler=-fopenmp
    LINK_LIBRARIES -fopenmp
    TEST_REGEX "PASS"
    TEST_ARGS 1
)
