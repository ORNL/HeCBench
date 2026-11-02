# kda-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME kda
    MODEL omp
    SOURCES main.cpp
    INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../kda-cuda
    CATEGORIES ml
    TEST_REGEX "PASS"
    TEST_ARGS 1 16 2
)
