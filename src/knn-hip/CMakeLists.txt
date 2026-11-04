# knn-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME knn
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
