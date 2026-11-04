# ddbp-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME ddbp
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES simulation
)
