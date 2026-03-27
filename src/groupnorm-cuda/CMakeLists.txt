# groupnorm-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME groupnorm
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
