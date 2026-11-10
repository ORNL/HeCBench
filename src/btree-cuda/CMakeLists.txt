# btree-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME btree
    MODEL cuda
    SOURCES main.cu
    CATEGORIES search
)
