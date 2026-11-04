# page-rank-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME page-rank
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
