# convolutionSeparable-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME convolutionSeparable
    MODEL cuda
    SOURCES conv.cu conv_gold.cu main.cu
    CATEGORIES imaging
)
