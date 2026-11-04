# p4-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME p4
    MODEL hip
    SOURCES main.cu
    COMPILE_OPTIONS -ffast-math
    CATEGORIES ml
)
