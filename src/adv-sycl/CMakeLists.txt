# adv-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME adv
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES simulation
    COMPILE_OPTIONS -Ddfloat=double -Ddlong=int
)
