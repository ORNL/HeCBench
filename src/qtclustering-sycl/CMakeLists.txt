# qtclustering-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME qtclustering
    MODEL sycl
    SOURCES Option.cpp OptionParser.cpp QTC.cpp comm.cpp libdata.cpp main.cpp
    CATEGORIES ml
)
