# convolutionSeparable-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME convolutionSeparable
    MODEL hip
    SOURCES conv.cu conv_gold.cu main.cu
    CATEGORIES imaging
)
