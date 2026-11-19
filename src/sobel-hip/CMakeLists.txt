# sobel-hip/CMakeLists.txt

REQUIREDATAFILE("${CMAKE_CURRENT_LIST_DIR}/../data/sobel/SobelFilter_Input.bmp")

set(SDK_DIR "${CMAKE_CURRENT_LIST_DIR}/../include")

add_hecbench_benchmark(
    NAME sobel
    MODEL hip
    SOURCES main.cu reference.cu
    INCLUDE_DIRS ${SDK_DIR}
    CATEGORIES imaging
)
