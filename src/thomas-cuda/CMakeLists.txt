# thomas-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME thomas
    MODEL cuda
    SOURCES cuThomasBatch.cu main.cu
    CATEGORIES math
)
