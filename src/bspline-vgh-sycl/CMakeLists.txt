# bspline-vgh-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME bspline-vgh
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES simulation
)
