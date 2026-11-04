# clenergy-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME clenergy
    MODEL hip
    SOURCES WKFUtils.cu clenergy.cu
    CATEGORIES simulation
)
