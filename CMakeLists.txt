cmake_minimum_required(VERSION 3.6)
project(nnlearn LANGUAGES CXX C)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fpermissive -fvisibility=hidden -Wall -std=c++17" )

set(CMAKE_CXX_STANDARD 14)

message(${CMAKE_CXX_COMPILER_ID})

add_subdirectory(dlib)


MESSAGE("make files end")




#target_link_libraries()