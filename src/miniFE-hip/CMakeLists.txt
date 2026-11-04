# miniFE-hip/CMakeLists.txt

execute_process(
    COMMAND /bin/bash -c "make common_files && make generate_info"
    RESULT_VARIABLE RESULT_CODE
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

add_hecbench_benchmark(
    NAME miniFE
    MODEL hip
    SOURCES src/main.cpp
            src/YAML_Doc.cpp
            src/YAML_Element.cpp
            utils/BoxPartition.cpp
            utils/param_utils.cpp
            utils/utils.cpp
            utils/mytimer.cpp
    COMPILE_OPTIONS -DMINIFE_SCALAR=double 
                    -DMINIFE_LOCAL_ORDINAL=int
                    -DMINIFE_GLOBAL_ORDINAL=int
                    -DMINIFE_RESTRICT=__restrict__
                    -DMINIFE_CSR_MATRIX
                    -DMINIFE_INFO=1
                    -DMINIFE_KERNELS=0
    INCLUDE_DIRS src utils fem
    CATEGORIES simulation
)

set_source_files_properties(src/main.cpp       PROPERTIES LANGUAGE HIP)
