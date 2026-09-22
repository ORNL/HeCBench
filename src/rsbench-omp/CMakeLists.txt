# rsbench-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME rsbench
    MODEL omp
    SOURCES init.c io.c main.c material.c simulation.c utils.c
    CATEGORIES algorithms
    LINK_LIBRARIES m
)
set_source_files_properties(init.c io.c main.c material.c simulation.c utils.c PROPERTIES LANGUAGE CXX)
