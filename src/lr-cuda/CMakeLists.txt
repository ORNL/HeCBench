# lr-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME lr
    MODEL cuda
    SOURCES linear_iter.cu linear_par.cu main.cu
    CATEGORIES ml
)
