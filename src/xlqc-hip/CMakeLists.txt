# xlqc-hip/CMakeLists.txt

set(SRC_DIR "${CMAKE_CURRENT_LIST_DIR}")
UNZIPFILE("${SRC_DIR}/example.tar.gz")

set(INC_DIR "${SRC_DIR}/../xlqc-cuda")

add_hecbench_benchmark(
    NAME xlqc
    MODEL hip
    SOURCES cuda_rys_dp.cu cuda_rys_sp.cu main.cu
            ${INC_DIR}/scf.cc
            ${INC_DIR}/basis.cc
            ${INC_DIR}/int_lib/crys.cc
            ${INC_DIR}/int_lib/cints.cc
    INCLUDE_DIRS ${INC_DIR} ${INC_DIR}/int_lib
    CATEGORIES algorithms
    COMPILE_OPTIONS -fgpu-rdc
    LINK_LIBRARIES -fgpu-rdc --hip-link Eigen3::Eigen
)
