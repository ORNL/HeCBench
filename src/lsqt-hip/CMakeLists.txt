# lsqt-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME lsqt
    MODEL hip
    SOURCES anderson.cu charge.cu hamiltonian.cu lsqt.cu main.cu model.cu model_general.cu model_lattice.cu sigma.cu vector.cu
    COMPILE_OPTIONS -ffast-math
    CATEGORIES simulation
)
