# srad-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME srad
    MODEL hip
    SOURCES main.cu
    CATEGORIES imaging
)
