# linearprobing-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME linearprobing
    MODEL cuda
    SOURCES linearprobing.cu main.cu test.cu
    CATEGORIES search
)
