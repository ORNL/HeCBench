# contract-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME contract
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES compression
)
