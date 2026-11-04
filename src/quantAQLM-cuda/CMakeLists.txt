# quantAQLM-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME quantAQLM
    MODEL cuda
    SOURCES main.cu
    CATEGORIES compression
)
