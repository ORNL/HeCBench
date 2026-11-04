# tqs-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME tqs
    MODEL omp
    SOURCES host_task.cpp kernel.cpp main.cpp
    CATEGORIES simulation
)
