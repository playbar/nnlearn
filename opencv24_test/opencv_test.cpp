#include <assert.h>
#include <iostream>
#include <string>
#include <fstream>

#include "opencv24_test.hpp"

int main()
{
	int ret = bfmatch_point_matching();

	if (ret == 0)
		fprintf(stdout, "========== test success ==========\n");
	else
		fprintf(stderr, "########## test fail ##########\n");

	return 0;
}

