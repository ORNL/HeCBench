# bsearch-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME bsearch
    MODEL cuda
    SOURCES main.cu
    CATEGORIES search
)
