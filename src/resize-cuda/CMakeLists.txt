# resize-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME resize
    MODEL cuda
    SOURCES main.cu
    CATEGORIES imaging
)
