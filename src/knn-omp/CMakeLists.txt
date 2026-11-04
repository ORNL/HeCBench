# knn-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME knn
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
