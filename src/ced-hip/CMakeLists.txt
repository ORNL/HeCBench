# ced-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME ced
    MODEL hip
    SOURCES kernel_reference.cu main.cu
    CATEGORIES imaging
    LINK_LIBRARIES pthread
)
