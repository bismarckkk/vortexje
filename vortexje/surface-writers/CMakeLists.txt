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

add_library(surface-writers OBJECT ${SRCS})
target_link_libraries(surface-writers PUBLIC HighFive ConfigProvider)
