# putget-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME putget
    MODEL cuda
    SOURCES main.cu
    CATEGORIES bandwidth
)
