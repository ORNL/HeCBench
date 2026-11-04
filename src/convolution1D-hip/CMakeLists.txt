# convolution1D-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME convolution1D
    MODEL hip
    SOURCES main.cu
    CATEGORIES imaging
)
