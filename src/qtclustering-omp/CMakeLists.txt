# qtclustering-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME qtclustering
    MODEL omp
    SOURCES Option.cpp OptionParser.cpp QTC.cpp comm.cpp libdata.cpp main.cpp
    CATEGORIES ml
)
