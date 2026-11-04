# hotspot3D-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME hotspot3D
    MODEL omp
    SOURCES 3D.cpp 3D_helper.cpp
    CATEGORIES simulation
)
