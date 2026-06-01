#!/bin/bash

# Change directory to .sh file location
cd $(dirname $0)

# CMake Build Type
BUILD_TYPE=Debug

# Configure project
cmake -S . -B build -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DGNEURO_BUILD_TESTS=1

# Build project
cmake --build build --config $BUILD_TYPE -j $((`nproc` - 1))
