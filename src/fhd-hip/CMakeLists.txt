# fhd-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME fhd
    MODEL hip
    SOURCES main.cu
    CATEGORIES simulation
)
