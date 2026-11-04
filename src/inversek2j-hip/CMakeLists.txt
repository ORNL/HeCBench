# inversek2j-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME inversek2j
    MODEL hip
    SOURCES main.cu
    CATEGORIES robotics
)
