# goulash-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME goulash
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES simulation
)
