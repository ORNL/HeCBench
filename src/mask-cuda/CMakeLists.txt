# mask-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME mask
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
