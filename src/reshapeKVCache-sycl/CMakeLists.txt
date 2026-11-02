# reshapeKVCache-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME reshapeKVCache
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES bandwidth
)
