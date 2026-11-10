# ace-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME ace
    MODEL hip
    SOURCES main.cu
    CATEGORIES simulation
)
