# xsbench-omp/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../xsbench-cuda")

add_hecbench_benchmark(
    NAME xsbench
    MODEL omp
    SOURCES Simulation.cpp
            ${INC_DIR}/Main.cu
            ${INC_DIR}/io.cu
            ${INC_DIR}/GridInit.cu
            ${INC_DIR}/Materials.cu
            ${INC_DIR}/XSutils.cu
    CATEGORIES simulation
)

set_source_files_properties(${INC_DIR}/Main.cu       PROPERTIES LANGUAGE CXX)
set_source_files_properties(${INC_DIR}/io.cu         PROPERTIES LANGUAGE CXX)
set_source_files_properties(${INC_DIR}/GridInit.cu   PROPERTIES LANGUAGE CXX)
set_source_files_properties(${INC_DIR}/Materials.cu  PROPERTIES LANGUAGE CXX)
set_source_files_properties(${INC_DIR}/XSutils.cu    PROPERTIES LANGUAGE CXX)
