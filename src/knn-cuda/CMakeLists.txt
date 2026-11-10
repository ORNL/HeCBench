# knn-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME knn
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
