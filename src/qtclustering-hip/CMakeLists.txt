# qtclustering-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME qtclustering
    MODEL hip
    SOURCES Option.cpp OptionParser.cpp QTC.cu comm.cpp libdata.cpp main.cpp
    CATEGORIES ml
)
