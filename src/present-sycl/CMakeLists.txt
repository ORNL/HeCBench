# present-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME present
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES cryptography
)
