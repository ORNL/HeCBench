# damage-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME damage
    MODEL hip
    SOURCES main.cu
    CATEGORIES simulation
)
