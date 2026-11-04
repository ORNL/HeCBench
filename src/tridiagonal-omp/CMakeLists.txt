# tridiagonal-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME tridiagonal
    MODEL omp
    SOURCES cmd_arg_reader.cpp main.cpp shrUtils.cpp
    CATEGORIES simulation
)
