# hotspot-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME hotspot
    MODEL hip
    SOURCES hotspot.cu
    CATEGORIES simulation
)
