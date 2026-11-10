# tsa-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME tsa
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES simulation
)
