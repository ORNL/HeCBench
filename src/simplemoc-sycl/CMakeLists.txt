# simplemoc-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME simplemoc
    MODEL sycl
    SOURCES init.cpp io.cpp main.cpp
    CATEGORIES simulation
)
