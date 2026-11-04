# mxfp4-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME mxfp4
    MODEL hip
    SOURCES main.cu
    CATEGORIES simulation
)
