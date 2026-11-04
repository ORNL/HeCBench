# burger-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME burger
    MODEL cuda
    SOURCES main.cu
    CATEGORIES simulation
)
