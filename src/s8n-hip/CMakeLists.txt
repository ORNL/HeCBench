# s8n-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME s8n
    MODEL hip
    SOURCES main.cu
    CATEGORIES search
)
