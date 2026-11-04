# keogh-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME keogh
    MODEL omp
    SOURCES main.cpp
    CATEGORIES search
)
