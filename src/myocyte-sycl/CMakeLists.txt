# myocyte-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME myocyte
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES simulation
)
