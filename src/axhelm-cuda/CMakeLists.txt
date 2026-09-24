# axhelm-cuda/CMakeLists.txt

# enable Fortran for BlasLapack
enable_language(Fortran)

# Shared DVC archive for all axhelm variants
set(DATA_DIR "${CMAKE_CURRENT_LIST_DIR}")
UNZIPFILE("${DATA_DIR}/BlasLapack.tar.gz")
FILE(GLOB FORTRAN_SOURCES "${DATA_DIR}/BlasLapack/*.f")
if(NOT FORTRAN_SOURCES)
    message(WARNING
        "Skipping axhelm-cuda: BlasLapack sources missing. "
        "From the repository root run: dvc pull src/axhelm-cuda/BlasLapack.tar.gz")
    return()
endif()
add_library(BlasLapack_cuda STATIC ${FORTRAN_SOURCES})
target_compile_options(BlasLapack_cuda PRIVATE -fPIE)

add_hecbench_benchmark(
    NAME axhelm
    MODEL cuda
    SOURCES main.cu meshBasis.cpp
    CATEGORIES algorithms
    COMPILE_OPTIONS -Ddfloat=float -Ddlong=int
    LINK_LIBRARIES BlasLapack_cuda
)
