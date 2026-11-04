# sortKV-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME sortKV
    MODEL hip
    SOURCES main.cu
    CATEGORIES sorting
)
