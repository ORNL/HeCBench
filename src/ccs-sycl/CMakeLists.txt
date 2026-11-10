# ccs-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME ccs
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES bioinformatics
)
