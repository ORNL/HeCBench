# matern-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME matern
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
