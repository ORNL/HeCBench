# simplemoc-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME simplemoc
    MODEL omp
    SOURCES init.cpp io.cpp main.cpp
    CATEGORIES simulation
)
