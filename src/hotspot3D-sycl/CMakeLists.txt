# hotspot3D-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME hotspot3D
    MODEL sycl
    SOURCES 3D.cpp 3D_helper.cpp
    CATEGORIES simulation
)
