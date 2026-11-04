# nlll-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME nlll
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
