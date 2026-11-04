# crs-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME crs
    MODEL sycl
    SOURCES GCRSMatrix.cpp galois.cpp jerasure.cpp main.cpp utils.cpp
    CATEGORIES encoding
)
