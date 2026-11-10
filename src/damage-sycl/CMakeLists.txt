# damage-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME damage
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES simulation
)
