# chemv-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME chemv
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES simulation
)
