# jacobian-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME jacobian
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
)
