# quantVLLM-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME quantVLLM
    MODEL hip
    SOURCES main.cu
    CATEGORIES compression
)
