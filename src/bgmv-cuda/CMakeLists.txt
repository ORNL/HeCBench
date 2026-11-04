# bgmv-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME bgmv
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
)
