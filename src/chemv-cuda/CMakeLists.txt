# chemv-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME chemv
    MODEL cuda
    SOURCES kernel.cu main.cu
    CATEGORIES simulation
)
