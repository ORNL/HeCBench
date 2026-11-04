# minisweep-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME minisweep
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES simulation
)
