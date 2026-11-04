# allreduce-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME allreduce
    MODEL hip
    SOURCES collectives.cu main.cu timer.cu
    CATEGORIES bandwidth
)
