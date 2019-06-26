CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

add_definitions(-DBOOST_THREAD_POSIX)
add_definitions(-DBOOST_THREAD_USES_GETPAGESIZE)

# support C++11
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
# support C++14, when gcc version > 5.1, use -std=c++14 instead of c++1y
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y")

IF(NOT CMAKE_BUILD_TYPE)
	SET(CMAKE_BUILD_TYPE "Release")
	SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O2")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2")
ELSE()
	SET(CMAKE_BUILD_TYPE "Debug")
	SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -Wall -O2")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -O2")
ENDIF()
MESSAGE(STATUS "cmake build type: ${CMAKE_BUILD_TYPE}")

MESSAGE(STATUS "cmake current source dir: ${CMAKE_CURRENT_SOURCE_DIR}")
SET(PATH_TEST_FILES ${CMAKE_CURRENT_SOURCE_DIR})
SET(PATH_SRC_GFLAGS_FILES 	${CMAKE_CURRENT_SOURCE_DIR}/../gflags)
SET(PATH_SRC_GLOG_FILES 	${CMAKE_CURRENT_SOURCE_DIR}/../glog)
SET(PATH_SRC_HDF5_FILES 	${CMAKE_CURRENT_SOURCE_DIR}/../hdf5)
SET(PATH_SRC_LEVELDB_FILES 	${CMAKE_CURRENT_SOURCE_DIR}/../leveldb)
SET(PATH_SRC_LMDB_FILES 	${CMAKE_CURRENT_SOURCE_DIR}/../lmdb)
SET(PATH_SRC_PROTOBUF_FILES ${CMAKE_CURRENT_SOURCE_DIR}/../protobuf)
SET(PATH_SRC_SNAPPY_FILES 	${CMAKE_CURRENT_SOURCE_DIR}/../snappy)
MESSAGE(STATUS "path test files: ${PATH_TEST_FILES}")


# head file search path
INCLUDE_DIRECTORIES(
	${PATH_TEST_FILES}
	${PROJECT_SOURCE_DIR}/opencv/include
	${PROJECT_SOURCE_DIR}/opencv/modules/calib3d/include
	${PROJECT_SOURCE_DIR}/opencv/modules/core/include
	${PROJECT_SOURCE_DIR}/opencv/modules/dnn/include
	${PROJECT_SOURCE_DIR}/opencv/modules/features2d/include
	${PROJECT_SOURCE_DIR}/opencv/modules/flann/include
	${PROJECT_SOURCE_DIR}/opencv/modules/highgui/include
	${PROJECT_SOURCE_DIR}/opencv/modules/imgcodecs/include
	${PROJECT_SOURCE_DIR}/opencv/modules/imgproc/include
	${PROJECT_SOURCE_DIR}/opencv/modules/ml/include
	${PROJECT_SOURCE_DIR}/opencv/modules/objdetect/include
	${PROJECT_SOURCE_DIR}/opencv/modules/photo/include
	${PROJECT_SOURCE_DIR}/opencv/modules/shape/include
	${PROJECT_SOURCE_DIR}/opencv/modules/stitching/include
	${PROJECT_SOURCE_DIR}/opencv/modules/superres/include
	${PROJECT_SOURCE_DIR}/opencv/modules/videoio/include
	${PROJECT_SOURCE_DIR}/opencv/modules/video/include
	${PROJECT_SOURCE_DIR}/opencv/modules/videostab/include
	${PROJECT_SOURCE_DIR}/boost
	${PROJECT_SOURCE_DIR}/openblas
	${PROJECT_SOURCE_DIR}/gflags
	${PROJECT_SOURCE_DIR}/glog/src
	${PATH_SRC_HDF5_FILES}
	${PATH_SRC_HDF5_FILES}/src
	${PATH_SRC_LEVELDB_FILES}/include
	${PATH_SRC_LMDB_FILES} # if include this directory, will have many boost errors, so make system lmdb.h file
	${PATH_SRC_PROTOBUF_FILES}/src
	${PATH_SRC_SNAPPY_FILES}
)

link_libraries(opencv_core)
link_libraries(opencv_imgproc)
link_libraries(opencv_imgcodecs)
link_libraries(opencv_highgui)
link_libraries(opencv_video)
link_libraries(opencv_videoio)
link_libraries(opencv_objdetect)
link_libraries(opencv_ml)
link_libraries(hdf5-static)
link_libraries(hdf5_cpp-static)
link_libraries(hdf5_hl-static)
link_libraries(hdf5_hl_cpp-static)
#link_libraries(openblas)

# find dep library
SET(DEP_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/build CACHE PATH "dep library path")
MESSAGE(STATUS "dep library dir: ${DEP_LIB_DIR}")
LINK_DIRECTORIES(${DEP_LIB_DIR})

# find boost
#FIND_PACKAGE(Boost REQUIRED)
#INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

# find openblas
#FIND_PACKAGE(OpenBLAS REQUIRED) # no FindOpenBLAS.cmake file
#INCLUDE_DIRECTORIES(${OpenBLAS_INCLUDE_DIRS})

# recursive query match files :*.cpp
FILE(GLOB_RECURSE TEST_CPP_LIST ${PATH_TEST_FILES}/*.cpp)
FILE(GLOB_RECURSE TEST_CC_LIST  ${PATH_TEST_FILES}/*.cc)
#MESSAGE(STATUS "test cpp list: ${TEST_CPP_LIST} ${TEST_C_LIST}")

# build executable program
ADD_EXECUTABLE(caffe_3rd_test ${TEST_CPP_LIST} ${TEST_CC_LIST})
# add dependent library: static and dynamic
TARGET_LINK_LIBRARIES(caffe_3rd_test
	gflags
	glog
	lmdb
	leveldb
	libprotobuf_caff
	libprotoc_caff
	snappy
	boost_all
	dl # hdf5: undefined reference to dlopen
	pthread
)

