# face-cuda/CMakeLists.txt

UNZIPFILE("${CMAKE_CURRENT_LIST_DIR}/dataset.tar.gz")

add_hecbench_benchmark(
    NAME face
    MODEL cuda
    SOURCES haar.cu image.cu main.cu rectangles.cu stdio-wrapper.cu
    CATEGORIES imaging
)
