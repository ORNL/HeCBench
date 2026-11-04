# colorwheel-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME colorwheel
    MODEL cuda
    SOURCES main.cu
    CATEGORIES imaging
)
