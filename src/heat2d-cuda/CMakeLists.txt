# heat2d-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME heat2d
    MODEL cuda
    SOURCES main.cu
    CATEGORIES simulation
)
