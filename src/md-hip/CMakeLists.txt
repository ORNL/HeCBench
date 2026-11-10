# md-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME md
    MODEL hip
    SOURCES main.cu
    CATEGORIES simulation
)
