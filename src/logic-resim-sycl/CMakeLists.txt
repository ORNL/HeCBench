# logic-resim-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME logic-resim
    MODEL sycl
    SOURCES
        Simulation/src/gate/DelayMgr.cpp
        Simulation/src/gate/Gate.cpp
        Simulation/src/gate/TruthTable.cpp
        Simulation/src/main/main.cpp
        Simulation/src/parser/interParser.cpp
        Simulation/src/parser/SAIF.cpp
        Simulation/src/parser/VCD.cpp
        Simulation/src/parser/vcdParser.cpp
        Simulation/src/sim/Simulator.cpp
        Simulation/src/util/util.cpp
        Simulation/src/wave/Signal.cpp
        Simulation/src/wave/Wire.cpp
        Simulation/src/wave/WireMgr.cpp
    INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/Simulation/src/gate
        ${CMAKE_CURRENT_SOURCE_DIR}/Simulation/src/parser
        ${CMAKE_CURRENT_SOURCE_DIR}/Simulation/src/sim
        ${CMAKE_CURRENT_SOURCE_DIR}/Simulation/src/util
        ${CMAKE_CURRENT_SOURCE_DIR}/Simulation/src/wave
    CATEGORIES simulation
)
