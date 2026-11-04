# cfd-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME cfd
    MODEL sycl
    SOURCES euler3d.cpp
    CATEGORIES simulation
)
