# deredundancy-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME deredundancy
    MODEL cuda
    SOURCES main.cu utils.cu
    CATEGORIES bioinformatics
)
