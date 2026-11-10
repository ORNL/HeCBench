# heun-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME heun
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
    LINK_LIBRARIES -ldnnl -pthread
    TEST_REGEX "PASS"
    TEST_ARGS 1 1 4 8 8 4
)
