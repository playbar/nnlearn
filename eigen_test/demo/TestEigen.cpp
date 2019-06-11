#include <iostream>
#include "funset.hpp"

int main(int argc, char* argv[])
{
	int ret = 0;
	ret = test_calcCovarMatrix();
	ret =  test_meanStdDev();
	ret = test_trace();
    test_pseudoinverse();
    test_SVD();
    test_eigenvalues_eigenvectors();
    test_norm();
    test_inverse_matrix();
    test_mat_determinant();
	ret = test_mat_transpose();
    test_eigen_base();
    test_matrix_inverse();

	if (ret == 0)
		fprintf(stdout, "========== test success ==========\n");
	else
		fprintf(stderr, "********** test fail: %d **********\n", ret);

	return 0;
}
