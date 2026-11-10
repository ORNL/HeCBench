# lda-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME lda
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
