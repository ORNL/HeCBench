# pathfinder-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME pathfinder
    MODEL hip
    SOURCES main.cu
    CATEGORIES simulation
)
