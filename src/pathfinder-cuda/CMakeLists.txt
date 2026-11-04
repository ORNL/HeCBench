# pathfinder-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME pathfinder
    MODEL cuda
    SOURCES main.cu
    CATEGORIES simulation
)
