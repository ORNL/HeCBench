# geodesic-sycl/CMakeLists.txt

UNZIPFILE("${CMAKE_CURRENT_LIST_DIR}/locations.tar.gz")

add_hecbench_benchmark(
    NAME geodesic
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES geoscience
)
