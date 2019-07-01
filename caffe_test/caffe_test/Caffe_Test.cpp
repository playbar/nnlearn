#include <iostream>
#include "funset.hpp"

int main()
{
	int ret = 0;
//    ret = cifar10_convert();
	ret = lenet_5_mnist_test();

	if (0 == ret)
        fprintf(stdout, "========== test success ==========\n");
	else
        fprintf(stderr, "########## test fail ##########\n");

	return 0;
}
