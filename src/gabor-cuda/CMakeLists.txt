# gabor-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME gabor
    MODEL cuda
    SOURCES main.cu
    CATEGORIES imaging
    COMPILE_OPTIONS -use_fast_math
)
