# stsg-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME stsg
    MODEL sycl
    SOURCES Filter.cpp main.cpp
    CATEGORIES geoscience
    LINK_LIBRARIES gdal
)
