# keogh-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME keogh
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES search
)
