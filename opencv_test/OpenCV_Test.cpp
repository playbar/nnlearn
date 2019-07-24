#include <assert.h>
#include <iostream>
#include <string>
#include <fstream>

#include "fbc_cv_funset.hpp"
#include "opencv_funset.hpp"

int main()
{
//	int ret = run_all_test();
	int ret = feature_point_matching();

	if (ret == 0)
		fprintf(stdout, "========== test success ==========\n");
	else
		fprintf(stderr, "########## test fail ##########\n");

	return 0;
}

