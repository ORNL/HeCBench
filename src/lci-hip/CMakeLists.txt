# lci-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME lci
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
)
