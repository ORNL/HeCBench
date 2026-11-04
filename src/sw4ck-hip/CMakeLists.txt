# sw4ck-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME sw4ck
    MODEL hip
    SOURCES curvilinear4sg.cu main.cu
    CATEGORIES simulation
)
