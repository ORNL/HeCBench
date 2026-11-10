# streamcluster-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME streamcluster
    MODEL cuda
    SOURCES streamcluster.cu
    CATEGORIES ml
)
