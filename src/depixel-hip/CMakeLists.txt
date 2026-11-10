# depixel-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME depixel
    MODEL hip
    SOURCES main.cu
    CATEGORIES imaging
)
