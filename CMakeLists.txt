cmake_minimum_required(VERSION 3.10)
project(vortexje)

include_directories(.)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(cmake/cpm.cmake)

if(NOT DEFINED VPM_SETUP)
    if (MSVC)
        add_definitions(-D_USE_MATH_DEFINES)

        # Download at: https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.12/hdf5-1.12.3/bin/windows/hdf5-1.12.3-Std-win10_64-vs17.zip
        set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};C:/Program Files/HDF_Group/HDF5/1.12.3/cmake")
    else()
        set(HDF5_VERSION 1.10.7)
    endif()
endif()

find_package(HDF5 ${HDF5_VERSION} REQUIRED)
find_package(OpenMP REQUIRED)

CPMFindPackage(
    NAME           eigen
    GIT_TAG        main
    GIT_REPOSITORY "http://git.bismarck.xyz/bismarck/eigen.git"
)
get_target_property(EigenPath eigen INTERFACE_INCLUDE_DIRECTORIES)
include_directories(${EigenPath})

CPMFindPackage(
    NAME           highfive
    VERSION        2.10.1
    GIT_REPOSITORY "https://github.com/BlueBrain/HighFive.git"
    OPTIONS        "HIGHFIVE_USE_BOOST OFF" "HIGHFIVE_USE_EIGEN OFF" "HIGHFIVE_STATIC_HDF5 ON" "HIGHFIVE_EXAMPLES OFF" "HIGHFIVE_BUILD_DOCS OFF" "HIGHFIVE_UNIT_TESTS OFF"
)
target_compile_definitions(HighFive INTERFACE HIGHFIVE_USE_EIGEN)

CPMFindPackage(
    NAME           rply
    GIT_TAG        main
    GIT_REPOSITORY "http://git.bismarck.xyz/bismarck/rply.git"
)

CPMFindPackage(
    NAME           ConfigProvider
    GIT_TAG        main
    GIT_REPOSITORY "http://git.bismarck.xyz/bismarck/ConfigProvider.git"
)

add_subdirectory(vortexje/boundary-layers)
add_subdirectory(vortexje/empirical-wakes)
add_subdirectory(vortexje/surface-loaders)
add_subdirectory(vortexje/surface-writers)
add_subdirectory(vortexje/field-writers)
add_subdirectory(vortexje/shape-generators)

set(SRCS
        vortexje/surface.cpp
        vortexje/solver.cpp
        vortexje/parameters.cpp
        vortexje/lifting-surface.cpp
        vortexje/wake.cpp
        vortexje/body.cpp
        vortexje/surface-builder.cpp
        vortexje/lifting-surface-builder.cpp
        vortexje/surface-writer.cpp
)

set(HDRS
        vortexje/surface.hpp
        vortexje/solver.hpp
        vortexje/parameters.hpp
        vortexje/lifting-surface.hpp
        vortexje/wake.hpp
        vortexje/boundary-layer.hpp
        vortexje/body.hpp
        vortexje/surface-builder.hpp
        vortexje/lifting-surface-builder.hpp
        vortexje/surface-loader.hpp
        vortexje/surface-writer.hpp
        vortexje/field-writer.hpp
)

add_library(
        vortexje STATIC ${SRCS}
)

target_link_libraries(
        vortexje PUBLIC
        boundary-layers
        empirical-wakes
        surface-loaders
        surface-writers
        field-writers
        shape-generators
        airfoils
        rply
        OpenMP::OpenMP_CXX
        HighFive
        eigen
        ConfigProvider
)

target_include_directories(vortexje PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if (MSVC)
    string(REPLACE "/RTC1" "" CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
    target_compile_options(vortexje PRIVATE /bigobj)
else ()
    target_compile_options(vortexje PRIVATE -march=native)
endif ()
