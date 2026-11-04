# linearprobing-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME linearprobing
    MODEL hip
    SOURCES linearprobing.cu main.cu test.cu
    CATEGORIES search
)
