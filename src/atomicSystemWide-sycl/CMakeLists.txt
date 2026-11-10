# atomicSystemWide-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME atomicSystemWide
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES compression
)
