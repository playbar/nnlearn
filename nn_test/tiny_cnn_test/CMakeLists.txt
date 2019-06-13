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

OPTION(USE_SSE "set to ON to use sse" OFF)
OPTION(USE_AVX "set to ON to use avx" OFF)
OPTION(USE_OMP "set to ON to use openmp" OFF)

IF(USE_SSE)
	ADD_DEFINITIONS(-DCNN_USE_SSE)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse3")
ENDIF()

IF(USE_AVX)
	ADD_DEFINITIONS(-DCNN_USE_AVX)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
ENDIF()

IF(USE_OMP)
	FIND_PACKAGE(OpenMP REQUIRED)
	ADD_DEFINITIONS(-DCNN_USE_OMP)
	SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF()

MESSAGE(STATUS "cmake current source dir: ${CMAKE_CURRENT_SOURCE_DIR}")
SET(PATH_TEST_FILES ${CMAKE_CURRENT_SOURCE_DIR}/)
SET(PATH_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/../tiny-cnn)
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
	${PATH_SRC_FILES}
)

link_libraries(opencv_core)
link_libraries(opencv_imgproc)
link_libraries(opencv_imgcodecs)
link_libraries(opencv_highgui)
link_libraries(opencv_video)
link_libraries(opencv_videoio)
link_libraries(opencv_objdetect)
link_libraries(opencv_ml)


# recursive query match files :*.cpp
FILE(GLOB_RECURSE TEST_CPP_LIST ${PATH_TEST_FILES}/*.cpp)
#MESSAGE(STATUS "test cpp list: ${TEST_CPP_LIST} ${TEST_C_LIST}")

# build executable program
ADD_EXECUTABLE(tiny_cnn_test ${TEST_CPP_LIST} ${TEST_C_LIST})
# add dependent library: static and dynamic
TARGET_LINK_LIBRARIES(tiny_cnn_test
		pthread)

