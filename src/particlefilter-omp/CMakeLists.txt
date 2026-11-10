# particlefilter-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME particlefilter
    MODEL omp
    SOURCES main.cpp
    CATEGORIES simulation
)
