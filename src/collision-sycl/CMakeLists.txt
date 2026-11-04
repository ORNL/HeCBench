# collision-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME collision
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES language
)
