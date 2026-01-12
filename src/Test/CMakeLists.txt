cmake_minimum_required(VERSION 3.15)
project(GNeuro-Test CXX)

file(GLOB SOURCES "*.cpp")
add_executable(GNeuro-Test ${SOURCES})
target_include_directories(GNeuro-Test PRIVATE ../../include)
target_link_libraries(GNeuro-Test PRIVATE GNeuro)
