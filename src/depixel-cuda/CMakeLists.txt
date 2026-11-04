# depixel-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME depixel
    MODEL cuda
    SOURCES main.cu
    CATEGORIES imaging
)
