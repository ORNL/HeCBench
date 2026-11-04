# segsort-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME segsort
    MODEL hip
    SOURCES main.cu
    CATEGORIES sorting
)
