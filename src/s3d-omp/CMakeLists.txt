# s3d-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME s3d
    MODEL omp
    SOURCES Option.cpp OptionParser.cpp S3D.cpp main.cpp
    CATEGORIES simulation
)
