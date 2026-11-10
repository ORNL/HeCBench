# ising-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME ising
    MODEL hip
    SOURCES main.cu
    CATEGORIES simulation
)
