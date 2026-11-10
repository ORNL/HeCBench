# cbsfil-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME cbsfil
    MODEL cuda
    SOURCES main.cu
    CATEGORIES imaging
)
