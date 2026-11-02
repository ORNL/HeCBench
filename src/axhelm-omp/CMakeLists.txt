# axhelm-omp/CMakeLists.txt

# Shared DVC archive lives in axhelm-cuda
set(DATA_DIR "${CMAKE_CURRENT_LIST_DIR}/../axhelm-cuda")
UNZIPFILE("${DATA_DIR}/BlasLapack.tar.gz")
# Configure-time glob: extracting the archive later stays invisible until CMake re-runs.
FILE(GLOB FORTRAN_SOURCES "${DATA_DIR}/BlasLapack/*.f")
if(NOT FORTRAN_SOURCES)
    message(WARNING
        "Skipping axhelm-omp: BlasLapack sources missing. "
        "From the repository root run: dvc pull src/axhelm-cuda/BlasLapack.tar.gz\n"
        "Then re-run CMake to register the benchmark; `cmake --build` alone will not pick it up.")
    return()
endif()

# BlasLapack is Fortran; require a Fortran compiler only once its sources are present
enable_language(Fortran)
add_library(BlasLapack STATIC ${FORTRAN_SOURCES})

add_hecbench_benchmark(
    NAME axhelm
    MODEL omp
    SOURCES main.cpp meshBasis.cpp
    CATEGORIES algorithms
    COMPILE_OPTIONS -Ddfloat=float -Ddlong=int
    LINK_LIBRARIES BlasLapack
)
