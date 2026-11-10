# sssp-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME sssp
    MODEL cuda
    SOURCES kernel.cu main.cu
    CATEGORIES graph
    LINK_LIBRARIES pthread
)
