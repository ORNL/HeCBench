# laplace-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME laplace
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES simulation
)
