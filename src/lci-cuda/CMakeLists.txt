# lci-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME lci
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
)
