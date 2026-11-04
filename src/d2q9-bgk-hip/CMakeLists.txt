# d2q9-bgk-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME d2q9-bgk
    MODEL hip
    SOURCES main.cu
    CATEGORIES simulation
)
