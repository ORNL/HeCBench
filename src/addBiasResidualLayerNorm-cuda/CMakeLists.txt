# addBiasResidualLayerNorm-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME addBiasResidualLayerNorm
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
