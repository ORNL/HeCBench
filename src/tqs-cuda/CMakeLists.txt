# tqs-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME tqs
    MODEL cuda
    SOURCES host_task.cpp kernel.cu main.cu
    CATEGORIES simulation
)
