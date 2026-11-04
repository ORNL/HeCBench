# multimaterial-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME multimaterial
    MODEL hip
    SOURCES compact.cu full_matrix.cu multimat.cu
    CATEGORIES simulation
)
