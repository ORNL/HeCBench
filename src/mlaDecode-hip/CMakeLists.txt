# mlaDecode-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME mlaDecode
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
