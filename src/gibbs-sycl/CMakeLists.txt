# gibbs-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME gibbs
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES simulation
)
