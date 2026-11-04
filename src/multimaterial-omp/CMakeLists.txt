# multimaterial-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME multimaterial
    MODEL omp
    SOURCES compact.cpp full_matrix.cpp multimat.cpp
    CATEGORIES simulation
)
