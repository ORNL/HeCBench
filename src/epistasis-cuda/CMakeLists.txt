# epistasis-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME epistasis
    MODEL cuda
    SOURCES main.cu
    CATEGORIES bioinformatics
)
