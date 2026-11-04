# minimod-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME minimod
    MODEL cuda
    SOURCES constants.cu data_setup.cu grid.cu main.cu minimig.cu pml.cu
    CATEGORIES simulation
)
