cmake_minimum_required(VERSION 3.15)
project(GNeuro CXX)

file(GLOB SOURCES "*.cpp")
add_library(GNeuro ${SOURCES})
target_include_directories(GNeuro PRIVATE ../../include)
target_include_directories(GNeuro PUBLIC  ../../external/GParsing/include/)
