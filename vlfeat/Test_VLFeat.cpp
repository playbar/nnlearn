#include <iostream>
#include "funset.hpp"

int main()
{
	VL_PRINT("Hello world!\n");
	int ret = 0;

	if (ret == 0)
		fprintf(stdout, "========== test success ==========\n");
	else
		fprintf(stderr, "########## test fail ##########\n");

	return 0;
}
