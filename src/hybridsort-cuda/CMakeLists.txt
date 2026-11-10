# hybridsort-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME hybridsort
    MODEL cuda
    SOURCES bucketsort.cu hybridsort.cu mergesort.cu
    CATEGORIES sorting
)
