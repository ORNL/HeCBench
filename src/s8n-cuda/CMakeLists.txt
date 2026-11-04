# s8n-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME s8n
    MODEL cuda
    SOURCES main.cu
    CATEGORIES search
)
