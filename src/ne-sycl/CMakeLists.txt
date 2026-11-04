# ne-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME ne
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES imaging
)
