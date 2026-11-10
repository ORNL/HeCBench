# epistasis-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME epistasis
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES bioinformatics
)
