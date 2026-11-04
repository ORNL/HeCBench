# memcpy-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME memcpy
    MODEL hip
    SOURCES main.cu
    CATEGORIES bandwidth
)
