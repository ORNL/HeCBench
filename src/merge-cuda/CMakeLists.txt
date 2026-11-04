# merge-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME merge
    MODEL cuda
    SOURCES main.cu
    CATEGORIES sorting
)
