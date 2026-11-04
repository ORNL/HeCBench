# wlcpow-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME wlcpow
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES simulation
)
