add_hecbench_benchmark(
    NAME abacus
    MODEL cuda
    SOURCES main.cpp ylm.cu neighborlist.cu reference.cpp
    CATEGORIES simulation
    TEST_ARGS 1024 4 4096 0.15 1
    TEST_REGEX "ABACUS: PASS"
)
