# che-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME che
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES simulation
)
