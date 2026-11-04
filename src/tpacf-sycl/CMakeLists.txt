# tpacf-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME tpacf
    MODEL sycl
    SOURCES args.cpp compute.cpp main.cpp
    CATEGORIES simulation
)
