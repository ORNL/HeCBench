# maxpool3d-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME maxpool3d
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
