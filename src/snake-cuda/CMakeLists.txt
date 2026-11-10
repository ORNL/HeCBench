# snake-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME snake
    MODEL cuda
    SOURCES main.cu
    CATEGORIES bioinformatics
)
