# tsp-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME tsp
    MODEL hip
    SOURCES main.cu
    COMPILE_OPTIONS -ffast-math
    CATEGORIES search
)
