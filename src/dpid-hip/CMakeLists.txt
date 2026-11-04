# dpid-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME dpid
    MODEL hip
    SOURCES kernels.cu main.cu
    CATEGORIES imaging
)
