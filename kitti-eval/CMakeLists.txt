cmake_minimum_required (VERSION 2.6)
project(devkit_object)

find_package(PNG REQUIRED)

add_executable(evaluate_object evaluate_object.cpp)
include_directories(${PNG_INCLUDE_DIR})
target_link_libraries(evaluate_object ${PNG_LIBRARY})
