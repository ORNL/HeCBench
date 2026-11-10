# inversek2j-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME inversek2j
    MODEL cuda
    SOURCES main.cu
    CATEGORIES robotics
)
