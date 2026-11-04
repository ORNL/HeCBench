# mallocFree-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME mallocFree
    MODEL cuda
    SOURCES main.cu
    CATEGORIES language
)
