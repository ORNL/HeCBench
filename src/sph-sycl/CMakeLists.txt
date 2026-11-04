# sph-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME sph
    MODEL sycl
    SOURCES fileio.cpp fluid.cpp geometry.cpp
    CATEGORIES simulation
)
