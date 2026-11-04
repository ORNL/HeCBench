# lif-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME lif
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
