# morphology-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME morphology
    MODEL sycl
    SOURCES main.cpp morphology.cpp
    CATEGORIES imaging
)
