# haccmk-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME haccmk
    MODEL omp
    SOURCES haccmk.cpp
    CATEGORIES simulation
)
