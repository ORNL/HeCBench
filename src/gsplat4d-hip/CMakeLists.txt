# gsplat4d-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME gsplat4d
    MODEL hip
    SOURCES main.cu
    CATEGORIES simulation
    TEST_ARGS 200000 1280 720 100
    TEST_REGEX "(?s)MPM step: PASS.*4D preprocess \([0-9]+ of [0-9]+ gaussians visible\): PASS.*Rasterizer: PASS"
    TEST_TIMEOUT 300
)
