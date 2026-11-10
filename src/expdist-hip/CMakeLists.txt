# expdist-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME expdist
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
