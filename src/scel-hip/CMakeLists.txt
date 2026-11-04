# scel-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME scel
    MODEL hip
    SOURCES main.cu
    COMPILE_OPTIONS -ffast-math
    CATEGORIES ml
)
