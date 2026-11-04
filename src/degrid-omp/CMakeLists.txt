# degrid-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME degrid
    MODEL omp
    SOURCES main.cpp
    CATEGORIES imaging
    COMPILE_OPTIONS -DPRECISION=double
)
