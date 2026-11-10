# knn-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME knn
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
