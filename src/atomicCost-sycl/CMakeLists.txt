# atomicCost-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME atomicCost
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES compression
)
