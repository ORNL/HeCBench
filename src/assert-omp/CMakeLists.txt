# assert-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME assert
    MODEL omp
    SOURCES main.cpp
    CATEGORIES language
)
