# cbsfil-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME cbsfil
    MODEL omp
    SOURCES main.cpp
    CATEGORIES imaging
)
