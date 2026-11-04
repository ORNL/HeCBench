# score-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME score
    MODEL cuda
    SOURCES main.cu
    CATEGORIES imaging
)
