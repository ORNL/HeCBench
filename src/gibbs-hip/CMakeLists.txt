# gibbs-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME gibbs
    MODEL hip
    SOURCES main.cu
    CATEGORIES simulation
)
