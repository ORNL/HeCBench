# dpid-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME dpid
    MODEL cuda
    SOURCES kernels.cu main.cu
    CATEGORIES imaging
)
