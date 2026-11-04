# stencil1d-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME stencil1d
    MODEL hip
    SOURCES stencil_1d.cu
    CATEGORIES imaging
)
