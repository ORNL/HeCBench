# vmc-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME vmc
    MODEL hip
    SOURCES vmc2.cu
    CATEGORIES simulation
)
