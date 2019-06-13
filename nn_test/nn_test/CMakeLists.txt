MESSAGE(STATUS "cmake current source dir: ${CMAKE_CURRENT_SOURCE_DIR}")
SET(PATH_TEST_FILES ${CMAKE_CURRENT_SOURCE_DIR}/)
SET(PATH_SRC_NN_FILES ${CMAKE_CURRENT_SOURCE_DIR}/../libnn)
SET(PATH_SRC_SVM_FILES ${CMAKE_CURRENT_SOURCE_DIR}/../libsvm)
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
	${PATH_SRC_NN_FILES}
	${PATH_SRC_SVM_FILES}
)

link_libraries(opencv_core)
link_libraries(opencv_imgproc)
link_libraries(opencv_imgcodecs)
link_libraries(opencv_highgui)
link_libraries(opencv_video)
link_libraries(opencv_videoio)
link_libraries(opencv_objdetect)
link_libraries(opencv_ml)

# find dep library
#SET(SEETAFACE_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/build CACHE PATH "seetaface library path")
#MESSAGE(STATUS "seetaface library dir: ${SEETAFACE_LIB_DIR}")
#LINK_DIRECTORIES(${SEETAFACE_LIB_DIR})

# recursive query match files :*.cpp
FILE(GLOB_RECURSE TEST_CPP_LIST ${PATH_TEST_FILES}/*.cpp)
FILE(GLOB_RECURSE SRC_NN_CPP_LIST ${PATH_SRC_NN_FILES}/*.cpp)
SET(SRC_SVM_CPP_LIST ${PATH_SRC_SVM_FILES}/svm.cpp)
#MESSAGE(STATUS "test cpp list: ${TEST_CPP_LIST} ${TEST_C_LIST}")

# build executable program
ADD_EXECUTABLE(nn_test ${TEST_CPP_LIST} ${SRC_NN_CPP_LIST} ${SRC_SVM_CPP_LIST})
# add dependent library: static and dynamic
TARGET_LINK_LIBRARIES(nn_test
		pthread)

