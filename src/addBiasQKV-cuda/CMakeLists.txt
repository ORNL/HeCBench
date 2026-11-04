# addBiasQKV-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME addBiasQKV
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
