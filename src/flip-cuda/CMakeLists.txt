# flip-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME flip
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
