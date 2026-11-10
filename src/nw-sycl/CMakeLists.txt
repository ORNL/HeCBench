# nw-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME nw
    MODEL sycl
    SOURCES nw.cpp
    CATEGORIES bioinformatics
)
