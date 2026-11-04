# lid-driven-cavity-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME lid-driven-cavity
    MODEL omp
    SOURCES main.cpp
    CATEGORIES simulation
)
