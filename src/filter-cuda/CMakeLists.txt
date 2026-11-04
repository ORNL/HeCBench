# filter-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME filter
    MODEL cuda
    SOURCES main.cu
    CATEGORIES compression
)
