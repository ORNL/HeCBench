# vanGenuchten-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME vanGenuchten
    MODEL cuda
    SOURCES main.cu
    CATEGORIES simulation
)
