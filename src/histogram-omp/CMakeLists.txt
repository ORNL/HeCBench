# histogram-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME histogram
    MODEL omp
    SOURCES histogram_compare_base.cpp
    CATEGORIES compression
)
