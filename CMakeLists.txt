cmake_minimum_required(VERSION 3.15)
project(GNeuro)
# enable_language(ASM_NASM)
# string(REPLACE "<CMAKE_ASM_NASM_COMPILER>" "<CMAKE_C_COMPILER>" CMAKE_ASM_NASM_LINK_EXECUTABLE ${CMAKE_ASM_NASM_LINK_EXECUTABLE})

add_library(GNeuro INTERFACE)
target_include_directories(GNeuro INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/external/GParsing/include/"
  "${CMAKE_CURRENT_SOURCE_DIR}/external/GMath/include/"
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_SHARED_LIBS OFF)
option(BUILD_STATIC_LIBS ON)

include(GNUInstallDirs)
set(CMAKE_INSTALL_RPATH "$ORIGIN/${CMAKE_INSTALL_LIBDIR} ${CMAKE_INSTALL_FULL_LIBDIR}")
message(STATUS "Using default install RPATH of: ${CMAKE_INSTALL_RPATH}")

file(GLOB SUBMODULES "external/*")

foreach(SUBMODULE ${SUBMODULES})
  if(IS_DIRECTORY ${SUBMODULE})
    if(EXISTS "${SUBMODULE}/CMakeLists.txt")
      message(STATUS "Adding submodule: ${SUBMODULE}")
      add_subdirectory(${SUBMODULE})
    endif()
  endif()
endforeach()

include("src/sources.cmake")
