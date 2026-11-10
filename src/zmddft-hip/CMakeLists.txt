# zmddft-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME zmddft
    MODEL hip
    SOURCES main.cu
    CATEGORIES signal
)
