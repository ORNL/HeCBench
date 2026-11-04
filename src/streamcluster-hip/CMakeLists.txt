# streamcluster-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME streamcluster
    MODEL hip
    SOURCES streamcluster.cu
    CATEGORIES ml
)
