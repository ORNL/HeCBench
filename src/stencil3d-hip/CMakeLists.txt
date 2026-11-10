# stencil3d-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME stencil3d
    MODEL hip
    SOURCES main.cu
    CATEGORIES imaging
)
