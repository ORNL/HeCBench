# pso-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME pso
    MODEL cuda
    SOURCES kernel_cpu.cpp kernel_gpu.cu main.cpp
    CATEGORIES simulation
)
