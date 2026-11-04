# loopback-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME loopback
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES simulation
)
