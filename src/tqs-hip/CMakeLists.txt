# tqs-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME tqs
    MODEL hip
    SOURCES host_task.cpp kernel.cu main.cu
    CATEGORIES simulation
)
