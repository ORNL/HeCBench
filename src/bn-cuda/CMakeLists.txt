# bn-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME bn
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
