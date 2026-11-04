# concat-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME concat
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
