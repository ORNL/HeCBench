# geodesic-omp/CMakeLists.txt

UNZIPFILE("${CMAKE_CURRENT_LIST_DIR}/../geodesic-sycl/locations.tar.gz")

add_hecbench_benchmark(
    NAME geodesic
    MODEL omp
    SOURCES main.cpp
    CATEGORIES geoscience
)
