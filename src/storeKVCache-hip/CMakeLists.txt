# storeKVCache-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME storeKVCache
    MODEL hip
    SOURCES main.cu
    CATEGORIES bandwidth
)
