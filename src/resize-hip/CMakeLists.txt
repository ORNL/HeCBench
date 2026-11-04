# resize-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME resize
    MODEL hip
    SOURCES main.cu
    CATEGORIES imaging
)
