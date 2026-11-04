# crossEntropy-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME crossEntropy
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
