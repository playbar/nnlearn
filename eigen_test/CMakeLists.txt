link_libraries(opencv_core)
link_libraries(opencv_imgproc)
link_libraries(opencv_imgcodecs)

SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -Wall -O2 -std=c11")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -g -Wall -O2 -std=c++11")
# 支持C++14, when gcc version > 5.1, use -std=c++14 instead of c++1y
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -g -Wall -O2 -std=c++1y")

MESSAGE(STATUS "project source dir: ${PROJECT_SOURCE_DIR}")
SET(PATH_SRC_FILES ${PROJECT_SOURCE_DIR}/eigen_test/demo)
MESSAGE(STATUS "path src files: ${PATH_SRC_FILES}")

# 指定头文件的搜索路径
INCLUDE_DIRECTORIES(
	${PATH_SRC_FILES}
	${PROJECT_SOURCE_DIR}/eigen_test/Eigen
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
)

# 递归查询所有匹配的文件:*.cpp
FILE(GLOB_RECURSE CPP_LIST ${PATH_SRC_FILES}/*.cpp)
FILE(GLOB_RECURSE C_LIST ${PATH_SRC_FILES}/*.c)
#MESSAGE(STATUS "cpp list: ${C_LIST}")

# 搜索指定的OpenCV外部依赖库头文件和库文件位置
#FIND_PACKAGE(OpenCV 2)

# 编译可执行程序
ADD_EXECUTABLE(Eigen_Test ${CPP_LIST} ${C_LIST})
# 用来为target添加需要链接的共享库，指定工程所用的依赖库，包括动态库和静态库
TARGET_LINK_LIBRARIES(Eigen_Test pthread)

