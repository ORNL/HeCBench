# asta-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME asta
    MODEL hip
    SOURCES main.cu
    CATEGORIES language
)
