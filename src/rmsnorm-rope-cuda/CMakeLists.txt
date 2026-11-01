# rmsnorm-rope-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME rmsnorm-rope
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
