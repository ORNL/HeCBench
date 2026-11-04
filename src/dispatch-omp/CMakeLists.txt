# dispatch-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME dispatch
    MODEL omp
    SOURCES main.cpp
    CATEGORIES language
)
