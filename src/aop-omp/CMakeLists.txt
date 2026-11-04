# aop-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME aop
    MODEL omp
    SOURCES main.cpp
    CATEGORIES finance
)
