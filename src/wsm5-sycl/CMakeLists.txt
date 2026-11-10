# wsm5-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME wsm5
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES simulation
    COMPILE_OPTIONS -DMKX=4
)
