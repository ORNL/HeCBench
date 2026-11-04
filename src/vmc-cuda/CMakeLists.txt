# vmc-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME vmc
    MODEL cuda
    SOURCES vmc2.cu
    CATEGORIES simulation
)
