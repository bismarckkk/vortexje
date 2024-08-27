include_directories(../)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

get_target_property(EigenPath eigen INTERFACE_INCLUDE_DIRECTORIES)
include_directories(${EigenPath})

add_subdirectory(boundary-layers)
add_subdirectory(empirical-wakes)
add_subdirectory(surface-loaders)
add_subdirectory(surface-writers)
add_subdirectory(field-writers)
add_subdirectory(shape-generators)

set(SRCS
        surface.cpp
        solver.cpp
        parameters.cpp
        lifting-surface.cpp
        wake.cpp
        body.cpp
        surface-builder.cpp
        lifting-surface-builder.cpp
        surface-writer.cpp)

set(HDRS
        surface.hpp
        solver.hpp
        parameters.hpp
        lifting-surface.hpp
        wake.hpp
        boundary-layer.hpp
        body.hpp
        surface-builder.hpp
        lifting-surface-builder.hpp
        surface-loader.hpp
        surface-writer.hpp
        field-writer.hpp)

add_library(
        vortexje ${SRCS}
)

target_link_libraries(
        vortexje
        boundary-layers
        empirical-wakes
        surface-loaders
        surface-writers
        field-writers
        shape-generators
        airfoils
        rply
        OpenMP::OpenMP_CXX
        eigen
        ConfigProvider
)

target_include_directories(vortexje PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)

if (MSVC)
    string(REPLACE "/RTC1" "" CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
    target_compile_options(vortexje PRIVATE /bigobj)
else ()
    target_compile_options(vortexje PRIVATE -march=native)
endif ()
