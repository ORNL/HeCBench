# sssp-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME sssp
    MODEL hip
    SOURCES kernel.cu main.cu
    CATEGORIES graph
    LINK_LIBRARIES pthread
)
