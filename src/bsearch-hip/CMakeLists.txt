# bsearch-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME bsearch
    MODEL hip
    SOURCES main.cu
    CATEGORIES search
)
