# affine-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME affine
    MODEL cuda
    SOURCES main.cu
    CATEGORIES imaging
)
