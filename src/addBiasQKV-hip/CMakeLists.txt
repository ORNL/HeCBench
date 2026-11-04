# addBiasQKV-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME addBiasQKV
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
