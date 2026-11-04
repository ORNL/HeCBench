# tridiagonal-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME tridiagonal
    MODEL hip
    SOURCES cmd_arg_reader.cu main.cu shrUtils.cu
    CATEGORIES simulation
)
