# GNeuro

## Table of contents:

1. [Description](#description)
2. [Installation](#installation)
3. [Usage](#usage)

## Description

### What is GNeuro?

A simple library to create and train neural networks of developer specified size.

### Why write my own library?

It is a simple library that I wrote to teach myself how neural networks work and to figure out
how backpropagation works on a fundamental level.

### Why use my own sub-library for JSON?

To keep the library under a permissive license that I control and to keep the project from scratch.

## Installation

To use this as a library simply download a copy from this repo and add it to your CMake project.

- Get a copy of the library:

  Use `git clone --recursive https://github.com/GustavJones/GNeuro.git`
  or `git submodule add https://github.com/GustavJones/GNeuro.git GNeuro` (recommended)
  or Download the ZIP archieve directly from Github.

- Add the library to the CMake project to build at compile time:

  Use `add_subdirectory()` in your CMake project and include GNeuro directory.

- Include and Link the library files to your project:

  Link the library code to your executable with `target_link_libraries()` with the library name **GNeuro**.
  [CMake Example](src/Test/CMakeLists.txt)

## Usage

> [!NOTE]
> An example can be seen in the [Test](src/Test)  source project included in this library.
