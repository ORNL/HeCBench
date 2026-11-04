# wsm5-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME wsm5
    MODEL omp
    SOURCES main.cpp
    CATEGORIES simulation
    COMPILE_OPTIONS -DMKX=4
)
