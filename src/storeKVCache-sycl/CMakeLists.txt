# storeKVCache-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME storeKVCache
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES bandwidth
)
