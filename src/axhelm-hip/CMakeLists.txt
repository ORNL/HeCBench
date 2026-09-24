# axhelm-hip/CMakeLists.txt

# enable Fortran for BlasLapack
enable_language(Fortran)

# Shared DVC archive lives in axhelm-cuda
set(DATA_DIR "${CMAKE_CURRENT_LIST_DIR}/../axhelm-cuda")
UNZIPFILE("${DATA_DIR}/BlasLapack.tar.gz")
FILE(GLOB FORTRAN_SOURCES "${DATA_DIR}/BlasLapack/*.f")
if(NOT FORTRAN_SOURCES)
    message(WARNING
        "Skipping axhelm-hip: BlasLapack sources missing. "
        "From the repository root run: dvc pull src/axhelm-cuda/BlasLapack.tar.gz")
    return()
endif()
add_library(BlasLapack_hip STATIC ${FORTRAN_SOURCES})

add_hecbench_benchmark(
    NAME axhelm
    MODEL hip
    SOURCES main.cu meshBasis.cpp
    CATEGORIES algorithms
    COMPILE_OPTIONS -Ddfloat=float -Ddlong=int
    LINK_LIBRARIES BlasLapack_hip
)
