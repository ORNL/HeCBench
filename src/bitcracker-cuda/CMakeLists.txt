# bitcracker-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME bitcracker
    MODEL cuda
    SOURCES attack.cu main.cu utils.cpp w_blocks.cu
    CATEGORIES algorithms
    COMPILE_OPTIONS -use_fast_math
)
