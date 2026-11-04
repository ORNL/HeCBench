# burger-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME burger
    MODEL hip
    SOURCES main.cu
    CATEGORIES simulation
)
