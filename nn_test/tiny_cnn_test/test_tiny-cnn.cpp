#include "funset.hpp"
#include <iostream>

int main()
{
	auto ret = test_tiny_cnn_predict();
	if (ret == 0) fprintf(stdout, "====== test success ======\n");
	else fprintf(stderr, "###### test fail ######");

	return 0;
}

