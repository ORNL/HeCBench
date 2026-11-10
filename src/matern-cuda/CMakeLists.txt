# matern-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME matern
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
