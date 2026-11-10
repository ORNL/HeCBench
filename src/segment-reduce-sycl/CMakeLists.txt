# segment-reduce-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME segment-reduce
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES compression
)
