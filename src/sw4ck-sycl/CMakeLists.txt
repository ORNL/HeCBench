# sw4ck-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME sw4ck
    MODEL sycl
    SOURCES curvilinear4sg.cpp main.cpp
    CATEGORIES simulation
)
