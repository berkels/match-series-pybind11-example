cmake_minimum_required(VERSION 2.8.12)
project(cmake_example)

if (NOT "$ENV{CONDA_PREFIX}" STREQUAL "")
    list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
ENDIF ()

FIND_PATH ( QUOCMESH_INCLUDE_DIR aol.h PATH_SUFFIXES quocmesh )
IF ( QUOCMESH_INCLUDE_DIR )
  INCLUDE_DIRECTORIES ( ${QUOCMESH_INCLUDE_DIR} )
ELSE ( )
  MESSAGE ( FATAL_ERROR "Could not find aol.h." )
ENDIF ( )

FIND_LIBRARY ( QUOCMESH_LIBRARY NAMES quocmesh )
IF ( NOT QUOCMESH_LIBRARY )
  MESSAGE ( FATAL_ERROR "Could not find quocmesh." )
ENDIF ()

find_package(pybind11)
pybind11_add_module(cmake_example src/main.cpp)

TARGET_LINK_LIBRARIES(cmake_example PRIVATE ${QUOCMESH_LIBRARY})
