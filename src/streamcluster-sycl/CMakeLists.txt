# streamcluster-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME streamcluster
    MODEL sycl
    SOURCES streamcluster.cpp
    CATEGORIES ml
)
