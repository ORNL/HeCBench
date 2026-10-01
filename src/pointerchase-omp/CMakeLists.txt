# pointerchase-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME pointerchase
    MODEL omp
    SOURCES main.cpp
    CATEGORIES algorithms
)
