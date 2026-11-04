# eikonal-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME eikonal
    MODEL cuda
    SOURCES StructuredEikonal.cu fim.cu kernel.cu main.cpp timer.cpp
    CATEGORIES simulation
)
