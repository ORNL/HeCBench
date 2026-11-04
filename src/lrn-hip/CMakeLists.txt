# lrn-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME lrn
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
