# geodesic-hip/CMakeLists.txt

UNZIPFILE("${CMAKE_CURRENT_LIST_DIR}/../geodesic-sycl/locations.tar.gz")

add_hecbench_benchmark(
    NAME geodesic
    MODEL hip
    SOURCES main.cu
    CATEGORIES geoscience
)
