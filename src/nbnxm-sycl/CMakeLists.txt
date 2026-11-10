# nbnxm-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME nbnxm
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES bioinformatics
)
