# hypterm-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME hypterm
    MODEL cuda
    SOURCES kernels.cu main.cpp reference.cpp
    CATEGORIES simulation
)
