# maxpool3d-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME maxpool3d
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
