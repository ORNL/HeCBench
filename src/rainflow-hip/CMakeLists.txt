# rainflow-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME rainflow
    MODEL hip
    SOURCES main.cu
    CATEGORIES simulation
)
