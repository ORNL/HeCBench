# heun-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME heun
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
    LINK_LIBRARIES -lMIOpen -pthread
    TEST_REGEX "PASS"
    TEST_ARGS 1 1 4 8 8 4
)
