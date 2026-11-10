# layout-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME layout
    MODEL cuda
    SOURCES main.cu
    CATEGORIES language
)
