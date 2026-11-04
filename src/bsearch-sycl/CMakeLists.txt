# bsearch-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME bsearch
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES search
)
