# tridiagonal-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME tridiagonal
    MODEL cuda
    SOURCES cmd_arg_reader.cu main.cu shrUtils.cu
    CATEGORIES simulation
    COMPILE_OPTIONS -DUNIX -DBENCH_ITERATIONS=100
)
