CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

# support C++11
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
# support C++14, when gcc version > 5.1, use -std=c++14 instead of c++1y
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y")

IF (BUILD_MODE STREQUAL "debug")
	MESSAGE(STATUS "****** BUILD MODE: DEBUG ******")
	SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -Wall -O2")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -g -Wall -O2")
ELSE()
	MESSAGE(STATUS "====== BUILD MODE: REALSE ======")
	SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O2")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2")
ENDIF()

MESSAGE(STATUS "project source dir: ${PROJECT_SOURCE_DIR}")
SET(PATH_TEST_FILES ${CMAKE_CURRENT_SOURCE_DIR})
MESSAGE(STATUS "path src files: ${PATH_TEST_FILES}")

# head file search path
INCLUDE_DIRECTORIES(
	${PATH_TEST_FILES}
	${PROJECT_SOURCE_DIR}/opencv24/include
	${PROJECT_SOURCE_DIR}/opencv24/modules/calib3d/include
	${PROJECT_SOURCE_DIR}/opencv24/modules/core/include
	${PROJECT_SOURCE_DIR}/opencv24/modules/dnn/include
	${PROJECT_SOURCE_DIR}/opencv24/modules/features2d/include
	${PROJECT_SOURCE_DIR}/opencv24/modules/flann/include
	${PROJECT_SOURCE_DIR}/opencv24/modules/highgui/include
	${PROJECT_SOURCE_DIR}/opencv24/modules/imgcodecs/include
	${PROJECT_SOURCE_DIR}/opencv24/modules/imgproc/include
	${PROJECT_SOURCE_DIR}/opencv24/modules/ml/include
	${PROJECT_SOURCE_DIR}/opencv24/modules/nonfree/include
	${PROJECT_SOURCE_DIR}/opencv24/modules/objdetect/include
	${PROJECT_SOURCE_DIR}/opencv24/modules/photo/include
	${PROJECT_SOURCE_DIR}/opencv24/modules/shape/include
	${PROJECT_SOURCE_DIR}/opencv24/modules/stitching/include
	${PROJECT_SOURCE_DIR}/opencv24/modules/superres/include
	${PROJECT_SOURCE_DIR}/opencv24/modules/videoio/include
	${PROJECT_SOURCE_DIR}/opencv24/modules/video/include
	${PROJECT_SOURCE_DIR}/opencv24/modules/videostab/include
)

# find library
link_libraries(opencv_core)
link_libraries(opencv_imgproc)
#link_libraries(opencv_imgcodecs)
link_libraries(opencv_features2d)
link_libraries(opencv_highgui)
link_libraries(opencv_video)
link_libraries(opencv_nonfree)
#link_libraries(opencv_videoio)
link_libraries(opencv_objdetect)

# recursive query match files :*.cpp, *c
FILE(GLOB_RECURSE TEST_CPP_LIST ${PATH_TEST_FILES}/*.cpp)
FILE(GLOB_RECURSE TEST_C_LIST   ${PATH_TEST_FILES}/*.c)
#MESSAGE(STATUS "cpp list: ${TEST_CPP_LIST} ${TEST_C_LIST}")

# build executable program
ADD_EXECUTABLE(opencv24_test ${TEST_CPP_LIST} ${TEST_C_LIST})
# add dependent library: static and dynamic
TARGET_LINK_LIBRARIES(opencv24_test
		pthread)

