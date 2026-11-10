# graphExecution-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME graphExecution
    MODEL cuda
    SOURCES main.cu
    CATEGORIES language
)
