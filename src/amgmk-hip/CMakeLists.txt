# amgmk-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME amgmk
    MODEL hip
    SOURCES csr_matrix.cu csr_matvec.cu hypre_error.cu hypre_memory.cu laplace.cu main.cu vector.cu
    CATEGORIES simulation
)
