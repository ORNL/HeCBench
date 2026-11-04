# sw4ck-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME sw4ck
    MODEL omp
    SOURCES main.cpp curvilinear4sg.cpp
    CATEGORIES simulation
)
