# filter-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME filter
    MODEL hip
    SOURCES main.cu
    CATEGORIES compression
)
