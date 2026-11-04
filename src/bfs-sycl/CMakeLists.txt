# bfs-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME bfs
    MODEL sycl
    SOURCES bfs.cpp
    CATEGORIES search
)
