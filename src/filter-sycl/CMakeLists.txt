# filter-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME filter
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES compression
)
