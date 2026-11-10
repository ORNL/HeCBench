# metropolis-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME metropolis
    MODEL hip
    SOURCES heap.cu main.cu utils.cu
    CATEGORIES simulation
)
