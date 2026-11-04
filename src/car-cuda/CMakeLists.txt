# car-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME car
    MODEL cuda
    SOURCES main.cu
    CATEGORIES imaging
)
