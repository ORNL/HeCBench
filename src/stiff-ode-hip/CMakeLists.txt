# stiff-ode-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME stiff-ode
    MODEL hip
    SOURCES main.cu
    CATEGORIES simulation
    COMPILE_OPTIONS -fopenmp
    LINK_LIBRARIES -fopenmp
    TEST_ARGS --zones 2 --iters 1 --steps 2 --warmup 0
    TEST_REGEX "stiff-ode: PASS"
    TEST_TIMEOUT 300
)
