#include <assert.h>
#include <iostream>
#include <string>
#include <fstream>

#include "opencv24_test.hpp"

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

