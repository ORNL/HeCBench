# degrid-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME degrid
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES imaging
    COMPILE_OPTIONS -DPRECISION=double
)
