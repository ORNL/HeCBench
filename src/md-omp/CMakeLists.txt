# md-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME md
    MODEL omp
    SOURCES main.cpp
    CATEGORIES simulation
)
