# segsort-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME segsort
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES sorting
)
