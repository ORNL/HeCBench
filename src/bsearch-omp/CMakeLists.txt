# bsearch-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME bsearch
    MODEL omp
    SOURCES main.cpp
    CATEGORIES search
)
