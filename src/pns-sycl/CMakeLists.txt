# pns-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME pns
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES simulation
)
