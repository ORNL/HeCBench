# adv-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME adv
    MODEL omp
    SOURCES main.cpp
    CATEGORIES simulation
    COMPILE_OPTIONS -Ddfloat=double -Ddlong=int
)
