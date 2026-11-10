# bitpacking-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME bitpacking
    MODEL cuda
    SOURCES kernels.cu main.cu utils.cu
    CATEGORIES compression
)
