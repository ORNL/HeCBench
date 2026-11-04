# tissue-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME tissue
    MODEL omp
    SOURCES main.cpp reference.cpp
    CATEGORIES simulation
)
