# rfs-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME rfs
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
)
