# ga-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME ga
    MODEL cuda
    SOURCES main.cu
    CATEGORIES bioinformatics
)
