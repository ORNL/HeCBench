# tqs-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME tqs
    MODEL sycl
    SOURCES host_task.cpp kernel.cpp main.cpp
    CATEGORIES simulation
)
