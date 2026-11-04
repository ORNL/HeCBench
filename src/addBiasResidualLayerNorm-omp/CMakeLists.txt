# addBiasResidualLayerNorm-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME addBiasResidualLayerNorm
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
