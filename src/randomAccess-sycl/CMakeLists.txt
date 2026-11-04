# randomAccess-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME randomAccess
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES bandwidth
)
