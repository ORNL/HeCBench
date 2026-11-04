# pool-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME pool
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
