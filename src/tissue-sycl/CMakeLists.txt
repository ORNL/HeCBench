# tissue-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME tissue
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES simulation
)
