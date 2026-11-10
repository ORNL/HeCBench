# lulesh-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME lulesh
    MODEL omp
    SOURCES lulesh-init.cc lulesh-util.cc lulesh-viz.cc lulesh.cc
    CATEGORIES simulation
)
