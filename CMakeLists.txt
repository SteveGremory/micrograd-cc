cmake_minimum_required(VERSION 3.22)
project(MicroGrad++)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(Main source/main.cc source/engine.cc source/nn.cc)
target_include_directories(Main PUBLIC source/)
