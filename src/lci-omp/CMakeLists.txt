# lci-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME lci
    MODEL omp
    SOURCES main.cpp
    CATEGORIES math
)
