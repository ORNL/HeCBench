# atomicReduction-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME atomicReduction
    MODEL sycl
    SOURCES reduction.cpp
    CATEGORIES compression
)
