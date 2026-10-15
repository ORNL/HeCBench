# stiff-ode-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME stiff-ode
    MODEL omp
    SOURCES main.cpp
    CATEGORIES simulation
    TEST_ARGS --zones 2 --iters 1 --steps 2 --warmup 0
    TEST_REGEX "stiff-ode: PASS"
    TEST_TIMEOUT 300
)
