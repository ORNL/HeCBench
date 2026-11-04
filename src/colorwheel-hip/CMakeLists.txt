# colorwheel-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME colorwheel
    MODEL hip
    SOURCES main.cu
    CATEGORIES imaging
)
