# match-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME match
    MODEL cuda
    SOURCES main.cu
    CATEGORIES imaging
)
