# sdma-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME sdma
    MODEL cuda
    SOURCES main.cu
    CATEGORIES bandwidth
    TEST_ARGS 2
    TEST_REGEX "PASS"
)
