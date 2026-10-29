add_hecbench_benchmark(
    NAME tbe
    MODEL hip
    SOURCES main.cu
    CATEGORIES machine-learning
    TEST_ARGS 2 64 1024 64 8 1
    COMPILE_OPTIONS -munsafe-fp-atomics
    INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR}/../tbe-cuda
)
