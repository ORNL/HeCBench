# hwt1d-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME hwt1d
    MODEL cuda
    SOURCES kernel.cu main.cu reference.cu
    CATEGORIES simulation
)
