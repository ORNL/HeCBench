# p2p-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME p2p
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES bandwidth
)
