# bgmv-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME bgmv
    MODEL omp
    SOURCES main.cpp
    CATEGORIES math
)
