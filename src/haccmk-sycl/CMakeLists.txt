# haccmk-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME haccmk
    MODEL sycl
    SOURCES haccmk.cpp
    CATEGORIES simulation
)
