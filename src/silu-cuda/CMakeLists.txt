# silu-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME silu
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
