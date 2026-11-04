# sampling-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME sampling
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
