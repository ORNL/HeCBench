# kalman-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME kalman
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
