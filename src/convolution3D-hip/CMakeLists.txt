# convolution3D-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME convolution3D
    MODEL hip
    SOURCES main.cu
    CATEGORIES imaging
)
