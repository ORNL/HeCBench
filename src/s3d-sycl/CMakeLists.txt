# s3d-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME s3d
    MODEL sycl
    SOURCES Option.cpp OptionParser.cpp S3D.cpp main.cpp
    CATEGORIES simulation
)
