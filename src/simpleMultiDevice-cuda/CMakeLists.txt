# simpleMultiDevice-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME simpleMultiDevice
    MODEL cuda
    SOURCES main.cu
    CATEGORIES language
)
