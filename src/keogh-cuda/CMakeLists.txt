# keogh-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME keogh
    MODEL cuda
    SOURCES main.cu
    CATEGORIES search
)
