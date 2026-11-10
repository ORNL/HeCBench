# stencil1d-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME stencil1d
    MODEL cuda
    SOURCES stencil_1d.cu
    CATEGORIES imaging
)
