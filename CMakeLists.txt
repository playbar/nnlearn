cmake_minimum_required(VERSION 3.6)
project(nnlearn LANGUAGES CXX C ASM)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fpermissive -fvisibility=hidden -Wall -std=c++17" )

set(CMAKE_CXX_STANDARD 14)

message(${CMAKE_CXX_COMPILER_ID})

add_subdirectory(openblas)
add_subdirectory(boost)
add_subdirectory(opencv)
add_subdirectory(caffe_test)
#add_subdirectory(cimg_test)
#add_subdirectory(dlib)
#add_subdirectory(eigen_test)
#add_subdirectory(nn_test)
#add_subdirectory(libyuv)
#add_subdirectory(simd_test)
#add_subdirectory(vigra)
#add_subdirectory(mxnet_test)

MESSAGE("make files end")




#target_link_libraries()