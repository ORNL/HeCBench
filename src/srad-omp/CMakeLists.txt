# srad-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME srad
    MODEL omp
    SOURCES main.c
            util/graphics/graphics.c
            util/graphics/resize.c
    CATEGORIES algorithms
    LINK_LIBRARIES m
)
set_source_files_properties(main.c util/graphics/graphics.c util/graphics/resize.c PROPERTIES LANGUAGE CXX)
