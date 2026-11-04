# qem-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME qem
    MODEL cuda
    SOURCES gpu_solver.cu main.cu
    CATEGORIES simulation
)
