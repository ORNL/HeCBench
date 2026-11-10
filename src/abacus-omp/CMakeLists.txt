set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../abacus-cuda")

add_hecbench_benchmark(
    NAME abacus
    MODEL omp
    SOURCES ${INC_DIR}/main.cpp ylm.cpp neighborlist.cpp ${INC_DIR}/reference.cpp
    CATEGORIES simulation
    TEST_ARGS 1024 4 4096 0.15 1
    TEST_REGEX "ABACUS: PASS"
)
