#include <iostream>
#include "funset.hpp"
#include "opencv.hpp"
#include "libsvm.hpp"

int main()
{
    int ret = 0;
    test_decision_tree_train();
//    test_pca();
//    test_libsvm_two_classification_train();
//	test_libsvm_two_classification_predict();
	
	if (ret == 0)
		fprintf(stdout, "========== test success ==========\n");
	else
		fprintf(stderr, "########## test fail ##########\n");

	return 0;
}

