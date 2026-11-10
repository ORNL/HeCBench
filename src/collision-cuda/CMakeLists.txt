# collision-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME collision
    MODEL cuda
    SOURCES main.cu
    CATEGORIES language
)
