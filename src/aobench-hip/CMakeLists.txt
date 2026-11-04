# aobench-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME aobench
    MODEL hip
    SOURCES ao.cu
    COMPILE_OPTIONS -ffast-math
    CATEGORIES imaging
)
