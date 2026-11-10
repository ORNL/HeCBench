# resnet-kernels-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME resnet-kernels
    MODEL hip
    SOURCES Kernel128_one.cu Kernel128_winograd.cu Kernel256_one.cu Kernel256_winograd.cu main.cu util.cu
    CATEGORIES ml
)
