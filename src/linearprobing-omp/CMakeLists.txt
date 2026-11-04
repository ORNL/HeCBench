# linearprobing-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME linearprobing
    MODEL omp
    SOURCES linearprobing.cpp main.cpp test.cpp
    CATEGORIES search
)
