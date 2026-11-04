# stencil3d-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME stencil3d
    MODEL cuda
    SOURCES main.cu
    CATEGORIES imaging
)
