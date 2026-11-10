# seam-carving-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME seam-carving
    MODEL hip
    SOURCES main.cu
    CATEGORIES imaging
)
