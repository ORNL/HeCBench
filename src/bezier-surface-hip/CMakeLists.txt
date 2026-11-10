# bezier-surface-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME bezier-surface
    MODEL hip
    SOURCES main.cu
    CATEGORIES imaging
)
