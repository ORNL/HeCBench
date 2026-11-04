# mnist-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME mnist
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
