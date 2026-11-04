# spm-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME spm
    MODEL hip
    SOURCES main.cu
    CATEGORIES imaging
)
