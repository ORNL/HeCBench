# reaction-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME reaction
    MODEL hip
    SOURCES main.cu
    CATEGORIES simulation
)
