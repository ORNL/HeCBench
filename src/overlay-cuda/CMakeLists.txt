# overlay-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME overlay
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
