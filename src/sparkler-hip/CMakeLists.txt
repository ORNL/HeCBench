# sparkler-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME sparkler
    MODEL hip
    SOURCES main.cu
    CATEGORIES simulation
    LINK_LIBRARIES hipblas
)
