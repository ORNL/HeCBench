add_hecbench_benchmark(
    NAME tbe
    MODEL omp
    SOURCES main.cpp
    CATEGORIES machine-learning
    TEST_ARGS 2 64 1024 64 8 1
    INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR}/../tbe-cuda
)
