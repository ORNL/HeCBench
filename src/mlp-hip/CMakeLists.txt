# mlp-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME mlp
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
    LINK_LIBRARIES hipblaslt
)
