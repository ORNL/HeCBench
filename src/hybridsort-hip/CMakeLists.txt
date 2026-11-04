# hybridsort-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME hybridsort
    MODEL hip
    SOURCES bucketsort.cu hybridsort.cu mergesort.cu
    CATEGORIES sorting
)
