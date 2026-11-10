# convolution3D-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME convolution3D
    MODEL cuda
    SOURCES main.cu
    CATEGORIES imaging
)
