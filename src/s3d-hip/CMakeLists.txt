# s3d-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME s3d
    MODEL hip
    SOURCES Option.cu OptionParser.cu S3D.cu main.cu
    CATEGORIES simulation
)
