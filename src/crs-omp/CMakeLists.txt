# crs-omp/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../crs-sycl")

add_hecbench_benchmark(
    NAME crs
    MODEL omp
    SOURCES  main.cpp utils.cpp
             ../crs-sycl/galois.cpp
             ../crs-sycl/jerasure.cpp
             ../crs-sycl/GCRSMatrix.cpp
    CATEGORIES encoding
    INCLUDE_DIRS ${INC_DIR}
)
