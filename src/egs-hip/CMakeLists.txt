# egs-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME egs
    MODEL hip
    SOURCES init.cu kernels.cu main.cu
    CATEGORIES imaging
)
