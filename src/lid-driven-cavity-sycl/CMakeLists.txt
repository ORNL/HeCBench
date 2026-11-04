# lid-driven-cavity-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME lid-driven-cavity
    MODEL sycl
    SOURCES main.cpp set.cpp
    CATEGORIES simulation
)
