# fhd-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME fhd
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES simulation
)
