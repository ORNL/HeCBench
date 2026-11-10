# lsqt-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME lsqt
    MODEL sycl
    SOURCES anderson.cpp charge.cpp hamiltonian.cpp lsqt.cpp main.cpp model.cpp model_general.cpp model_lattice.cpp sigma.cpp vector.cpp
    CATEGORIES simulation
    COMPILE_OPTIONS -ffast-math
)
