# affine-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME affine
    MODEL hip
    SOURCES main.cu
    CATEGORIES imaging
)
