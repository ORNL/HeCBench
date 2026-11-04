# pnpoly-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME pnpoly
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
)
