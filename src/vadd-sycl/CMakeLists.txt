# vadd-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME vadd
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES bandwidth
)
