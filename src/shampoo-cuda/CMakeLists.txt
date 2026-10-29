# shampoo-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME shampoo
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
    COMPILE_OPTIONS -Xcompiler=-fopenmp
    LINK_LIBRARIES cublas cusolver -fopenmp
    TEST_ARGS 33 31 3 1
    TEST_REGEX "PASS"
)
