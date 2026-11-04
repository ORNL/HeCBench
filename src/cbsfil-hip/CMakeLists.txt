# cbsfil-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME cbsfil
    MODEL hip
    SOURCES main.cu
    CATEGORIES imaging
)
