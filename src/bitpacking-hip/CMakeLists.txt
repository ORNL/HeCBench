# bitpacking-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME bitpacking
    MODEL hip
    SOURCES kernels.cu main.cu utils.cu
    CATEGORIES compression
)
