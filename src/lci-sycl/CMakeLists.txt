# lci-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME lci
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
)
