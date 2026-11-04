# minisweep-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME minisweep
    MODEL omp
    SOURCES main.cpp
    CATEGORIES simulation
)
