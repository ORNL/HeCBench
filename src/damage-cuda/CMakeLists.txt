# damage-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME damage
    MODEL cuda
    SOURCES main.cu
    CATEGORIES simulation
)
