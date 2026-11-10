# atomicAggregate-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME atomicAggregate
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES compression
)
