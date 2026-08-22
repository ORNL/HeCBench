# gsplat4d-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME gsplat4d
    MODEL cuda
    SOURCES main.cu
    CATEGORIES simulation
    TEST_ARGS 200000 1280 720 100
    TEST_REGEX "Rasterizer: PASS"
    TEST_TIMEOUT 300
)
