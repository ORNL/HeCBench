# silu-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME silu
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
