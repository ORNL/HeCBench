# particlefilter-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME particlefilter
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES simulation
)
