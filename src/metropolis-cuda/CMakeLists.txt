# metropolis-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME metropolis
    MODEL cuda
    SOURCES heap.cu main.cu utils.cu
    CATEGORIES simulation
)
