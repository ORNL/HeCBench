# sdma-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME sdma
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES bandwidth
    TEST_ARGS 2
    TEST_REGEX "PASS"
)
