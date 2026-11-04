# quantAQLM-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME quantAQLM
    MODEL hip
    SOURCES main.cu
    CATEGORIES compression
)
