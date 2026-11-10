# bezier-surface-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME bezier-surface
    MODEL omp
    SOURCES main.cpp
    CATEGORIES imaging
)
