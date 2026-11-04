# tsp-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME tsp
    MODEL cuda
    SOURCES main.cu
    CATEGORIES search
    COMPILE_OPTIONS -use_fast_math
)
