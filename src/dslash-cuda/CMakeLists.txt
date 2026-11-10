# dslash-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME dslash
    MODEL cuda
    SOURCES dslash.cu kernels.cu main.cu
    CATEGORIES simulation
    COMPILE_OPTIONS -use_fast_math
)
