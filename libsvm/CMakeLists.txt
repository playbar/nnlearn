#cnn

include_directories( ${CMAKE_CURRENT_SOURCE_DIR})
add_library(libsvm
        svm.cpp)

link_libraries(libsvm)

add_executable(svm_predict
        svm-predict.c)

add_executable(svm_scale
        svm-scale.c)

add_executable(svm_train
        svm-train.c)




