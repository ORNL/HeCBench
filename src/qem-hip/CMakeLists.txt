# qem-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME qem
    MODEL hip
    SOURCES gpu_solver.cu main.cu
    CATEGORIES simulation
)
