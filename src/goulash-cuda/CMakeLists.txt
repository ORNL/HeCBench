# goulash-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME goulash
    MODEL cuda
    SOURCES main.cu
    CATEGORIES simulation
)
