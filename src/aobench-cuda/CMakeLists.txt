# aobench-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME aobench
    MODEL cuda
    SOURCES ao.cu
    CATEGORIES imaging
    COMPILE_OPTIONS -use_fast_math
)
