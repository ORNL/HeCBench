# xlqc-sycl/CMakeLists.txt

set(SRC_DIR "${CMAKE_CURRENT_LIST_DIR}")
UNZIPFILE("${SRC_DIR}/example.tar.gz")

set(INC_DIR "${SRC_DIR}/../xlqc-cuda")

add_hecbench_benchmark(
    NAME xlqc
    MODEL sycl
    SOURCES cuda_rys_dp.cpp cuda_rys_sp.cpp main.cpp 
            ${INC_DIR}/scf.cc
            ${INC_DIR}/basis.cc
            ${INC_DIR}/int_lib/crys.cc
            ${INC_DIR}/int_lib/cints.cc
    INCLUDE_DIRS ${INC_DIR} ${INC_DIR}/int_lib
    CATEGORIES algorithms
    LINK_LIBRARIES Eigen3::Eigen
)
