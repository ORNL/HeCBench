# mdh-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME mdh
    MODEL omp
    SOURCES WKFUtils.cpp main.cpp
    CATEGORIES simulation
)
