# fpdc-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME fpdc
    MODEL hip
    SOURCES main.cu
    CATEGORIES compression
)
