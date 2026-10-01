# mlaDecode-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME mlaDecode
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
