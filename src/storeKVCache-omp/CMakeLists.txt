# storeKVCache-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME storeKVCache
    MODEL omp
    SOURCES main.cpp
    CATEGORIES bandwidth
)
