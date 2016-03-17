cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(main1)

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (main1 main1.cpp)
target_link_libraries (main1 ${PCL_LIBRARIES})
