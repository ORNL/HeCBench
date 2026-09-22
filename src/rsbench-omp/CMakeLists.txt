# rsbench-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME rsbench
    MODEL omp
    SOURCES init.c io.c main.c material.c simulation.c utils.c
    CATEGORIES algorithms
    LINK_LIBRARIES m
    C_AS_CXX
)
