# laplace3d-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME laplace3d
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES simulation
)
