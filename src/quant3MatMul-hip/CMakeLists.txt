# quant3MatMul-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME quant3MatMul
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
)
