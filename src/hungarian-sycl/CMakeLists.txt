# hungarian-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME hungarian
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES graph
)
