# ert-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME ert
    MODEL hip
    SOURCES main.cu
    CATEGORIES language
)
