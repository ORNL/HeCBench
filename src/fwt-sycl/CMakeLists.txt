# fwt-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME fwt
    MODEL sycl
    SOURCES main.cpp  reference.cpp
    CATEGORIES math
)
