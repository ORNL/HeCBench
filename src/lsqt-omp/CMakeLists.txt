# lsqt-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME lsqt
    MODEL omp
    SOURCES anderson.cpp charge.cpp hamiltonian.cpp lsqt.cpp main.cpp model.cpp model_general.cpp model_lattice.cpp sigma.cpp vector.cpp
    CATEGORIES simulation
)
