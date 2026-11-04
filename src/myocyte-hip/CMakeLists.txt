# myocyte-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME myocyte
    MODEL hip
    SOURCES main.cu
    CATEGORIES simulation
)
