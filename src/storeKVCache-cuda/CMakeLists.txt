# storeKVCache-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME storeKVCache
    MODEL cuda
    SOURCES main.cu
    CATEGORIES bandwidth
)
