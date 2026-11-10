# accuracy-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME accuracy
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
