# tensorAccessor-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME tensorAccessor
    MODEL cuda
    SOURCES main.cu
    CATEGORIES language
)
