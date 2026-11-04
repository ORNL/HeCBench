# minibude-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME minibude
    MODEL cuda
    SOURCES kernel.cu main.cu
    CATEGORIES bioinformatics
)
