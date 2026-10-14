# reshapeKVCache-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME reshapeKVCache
    MODEL cuda
    SOURCES main.cu
    CATEGORIES bandwidth
)
