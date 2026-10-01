# muon-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME muon
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
    LINK_LIBRARIES hipblas
)
