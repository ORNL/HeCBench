# pointerchase-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME pointerchase
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES language
)
