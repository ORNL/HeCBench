# rainflow-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME rainflow
    MODEL omp
    SOURCES main.cpp
    CATEGORIES simulation
)
