# lavaMD-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME lavaMD
    MODEL omp
    SOURCES main.cpp ./util/num/num.cpp
    CATEGORIES simulation
)
