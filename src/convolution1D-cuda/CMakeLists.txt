# convolution1D-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME convolution1D
    MODEL cuda
    SOURCES main.cu
    CATEGORIES imaging
)
