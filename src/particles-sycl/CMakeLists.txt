# particles-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME particles
    MODEL sycl
    SOURCES bitonicSort.cpp main.cpp particles.cpp
    CATEGORIES simulation
)
