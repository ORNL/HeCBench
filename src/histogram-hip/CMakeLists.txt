# histogram-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME histogram
    MODEL hip
    SOURCES histogram_compare.cu
    CATEGORIES compression
)
