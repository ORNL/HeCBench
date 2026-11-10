# hotspot-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME hotspot
    MODEL sycl
    SOURCES hotspot.cpp
    CATEGORIES simulation
)
