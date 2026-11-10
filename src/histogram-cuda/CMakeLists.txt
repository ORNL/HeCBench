# histogram-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME histogram
    MODEL cuda
    SOURCES histogram_compare.cu
    CATEGORIES compression
)
