# kda-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME kda
    MODEL hip
    SOURCES main.cu
    INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../kda-cuda
    CATEGORIES ml
    COMPILE_OPTIONS -fopenmp
    LINK_LIBRARIES -fopenmp
    TEST_REGEX "PASS"
    TEST_ARGS 1 16 2
)
