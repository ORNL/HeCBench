# qkv-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME qkv
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
    LINK_LIBRARIES hipblas hipblaslt
)
