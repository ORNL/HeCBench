# nonzero-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME nonzero
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
