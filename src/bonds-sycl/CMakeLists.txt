# bonds-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME bonds
    MODEL sycl
    SOURCES bondsEngine.cpp
    CATEGORIES finance
)
