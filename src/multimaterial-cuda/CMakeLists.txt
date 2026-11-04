# multimaterial-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME multimaterial
    MODEL cuda
    SOURCES compact.cu full_matrix.cu multimat.cu
    CATEGORIES simulation
)
