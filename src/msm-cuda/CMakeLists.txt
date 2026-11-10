# msm-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME msm
    MODEL cuda
    SOURCES main.cu
    CATEGORIES cryptography
    TEST_REGEX "PASS"
    TEST_ARGS 1 12
)
