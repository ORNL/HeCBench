# perlin-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME perlin
    MODEL cuda
    SOURCES main.cu noise.cu
    CATEGORIES imaging
)
