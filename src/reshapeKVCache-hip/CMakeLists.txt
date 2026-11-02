# reshapeKVCache-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME reshapeKVCache
    MODEL hip
    SOURCES main.cu
    CATEGORIES bandwidth
)
