# histogram-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME histogram
    MODEL sycl
    SOURCES histogram_compare_base.cpp
    CATEGORIES compression
)
