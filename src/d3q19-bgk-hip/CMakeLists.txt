# d3q19-bgk-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME d3q19-bgk
    MODEL hip
    SOURCES main.cu
    CATEGORIES simulation
)
