# cm-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME cm
    MODEL cuda
    SOURCES io.cpp main.cpp process.cu
    CATEGORIES bioinformatics
)
