cmake_minimum_required(VERSION 3.15)
project(GNeuro-Test CXX)

if(GNEURO_BUILD_TESTS)
  file(GLOB SOURCES "*.cpp")
  add_executable(GNeuro-Test ${SOURCES})
  target_link_libraries(GNeuro-Test PRIVATE GNeuro)
endif()
