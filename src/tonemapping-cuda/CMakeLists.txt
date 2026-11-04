# tonemapping-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME tonemapping
    MODEL cuda
    SOURCES main.cu
    CATEGORIES imaging
)
