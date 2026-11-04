# p4-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME p4
    MODEL omp
    SOURCES main.cpp
    ENABLE_FASTMATH true
    CATEGORIES ml
)
