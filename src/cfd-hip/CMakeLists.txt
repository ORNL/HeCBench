# cfd-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME cfd
    MODEL hip
    SOURCES euler3d.cu
    CATEGORIES simulation
)
