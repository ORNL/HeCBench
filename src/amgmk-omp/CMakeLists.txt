# amgmk-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME amgmk
    MODEL omp
    SOURCES csr_matrix.cpp csr_matvec.cpp hypre_error.cpp hypre_memory.cpp laplace.cpp main.cpp vector.cpp
    CATEGORIES simulation
)
