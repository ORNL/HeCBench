# hybridsort-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME hybridsort
    MODEL sycl
    SOURCES bucketsort.c hybridsort.c mergesort.c
    CATEGORIES sorting
    C_AS_CXX
)
