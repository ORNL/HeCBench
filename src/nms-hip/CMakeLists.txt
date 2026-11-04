# nms-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME nms
    MODEL hip
    SOURCES main.cu
    CATEGORIES imaging
)
