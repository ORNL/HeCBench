# heun-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME heun
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
    LINK_LIBRARIES -lcudnn -lpthread
    TEST_REGEX "PASS"
    TEST_ARGS 1 1 4 8 8 4
)
