# gemv-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME gemv
    MODEL sycl
    SOURCES gemv.cpp main.cpp
    CATEGORIES math
)
