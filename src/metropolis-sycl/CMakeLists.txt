# metropolis-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME metropolis
    MODEL sycl
    SOURCES heap.cpp main.cpp utils.cpp
    CATEGORIES simulation
)
