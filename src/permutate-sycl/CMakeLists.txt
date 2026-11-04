# permutate-sycl/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../permutate-cuda")

add_hecbench_benchmark(
    NAME permutate
    MODEL sycl
    SOURCES gpu_permutation_testing.cpp
            ${INC_DIR}/main.cpp
            ${INC_DIR}/permutation_testing.cpp
            ${INC_DIR}/utils.cpp
            ${INC_DIR}/statistical_test.cpp
            ${INC_DIR}/bzip/blocksort.cpp
            ${INC_DIR}/bzip/bzlib.cpp
            ${INC_DIR}/bzip/compress.cpp
            ${INC_DIR}/bzip/crctable.cpp
            ${INC_DIR}/bzip/decompress.cpp
            ${INC_DIR}/bzip/huffman.cpp
            ${INC_DIR}/bzip/randtable.cpp
    CATEGORIES rng
    INCLUDE_DIRS ${INC_DIR}

)
