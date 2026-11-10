# unfold-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME unfold
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
    COMPILE_OPTIONS --extended-lambda
)
