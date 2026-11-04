# vanGenuchten-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME vanGenuchten
    MODEL hip
    SOURCES main.cu
    CATEGORIES simulation
)
