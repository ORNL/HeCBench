# sheath-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME sheath
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES simulation
)
