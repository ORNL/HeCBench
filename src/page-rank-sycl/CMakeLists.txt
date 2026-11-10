# page-rank-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME page-rank
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
