# hybridsort-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME hybridsort
    MODEL omp
    SOURCES bucketsort.c hybridsort.c mergesort.c
    CATEGORIES algorithms
    C_AS_CXX
)
