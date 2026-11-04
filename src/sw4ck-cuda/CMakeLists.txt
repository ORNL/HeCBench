# sw4ck-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME sw4ck
    MODEL cuda
    SOURCES curvilinear4sg.cu main.cu
    CATEGORIES simulation
)
