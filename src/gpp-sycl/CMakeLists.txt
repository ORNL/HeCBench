# gpp-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME gpp
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES simulation
)
