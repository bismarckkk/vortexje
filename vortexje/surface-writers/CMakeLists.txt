set(
        SRCS
        gmsh-surface-writer.cpp
        vtk-surface-writer.cpp
        xdmf-surface-writer.cpp
)

set(
        HDRS
        gmsh-surface-writer.hpp
        vtk-surface-writer.hpp
        xdmf-surface-writer.hpp
)

find_package(HDF5 1.12.3 REQUIRED)

add_library(surface-writers OBJECT ${SRCS})
target_link_libraries(surface-writers PUBLIC ${HDF5_LIBRARIES} ConfigProvider)
target_include_directories(surface-writers PUBLIC ../../../highfive/include ${HDF5_INCLUDE_DIR})

install(FILES ${HDRS} DESTINATION include/vortexje/surface-writers)
