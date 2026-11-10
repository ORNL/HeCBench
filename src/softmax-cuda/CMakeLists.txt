# softmax-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME softmax
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
    COMPILE_OPTIONS -use_fast_math
)
