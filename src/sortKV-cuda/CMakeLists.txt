# sortKV-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME sortKV
    MODEL cuda
    SOURCES main.cu
    CATEGORIES sorting
)
