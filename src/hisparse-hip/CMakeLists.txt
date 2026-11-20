# hisparse-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME hisparse
    MODEL hip
    SOURCES main.cu hisparse_kernels.cu
    CATEGORIES ml
    TEST_ARGS 8 256 1
    TEST_REGEX "PASS"
)
