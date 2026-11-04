# ne-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME ne
    MODEL omp
    SOURCES main.cpp
    CATEGORIES imaging
)
