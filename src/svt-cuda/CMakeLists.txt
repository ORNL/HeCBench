# svt-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME svt
    MODEL cuda
    SOURCES main.cu
    CATEGORIES simulation
    TEST_ARGS 12 1280 720 100
    TEST_REGEX "Raycast: PASS"
    TEST_TIMEOUT 300
)
