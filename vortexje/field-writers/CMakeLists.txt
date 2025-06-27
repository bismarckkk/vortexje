set(SRCS
    vtk-field-writer.cpp)
	
set(HDRS
    vtk-field-writer.hpp)

add_library(field-writers OBJECT ${SRCS})
