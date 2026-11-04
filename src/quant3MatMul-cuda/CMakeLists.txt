# quant3MatMul-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME quant3MatMul
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
)
