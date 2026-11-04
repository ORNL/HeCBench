# fpdc-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME fpdc
    MODEL cuda
    SOURCES main.cu
    CATEGORIES compression
)
