# addBiasResidualLayerNorm-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME addBiasResidualLayerNorm
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
