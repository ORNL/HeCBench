# page-rank-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME page-rank
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
