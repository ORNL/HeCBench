# s3d-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME s3d
    MODEL cuda
    SOURCES Option.cu OptionParser.cu S3D.cu main.cu
    CATEGORIES simulation
)
