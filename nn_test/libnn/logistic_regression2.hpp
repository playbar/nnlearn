#ifndef FBC_SRC_NN_LOGISTIC_REGRESSION2_HPP_
#define FBC_SRC_NN_LOGISTIC_REGRESSION2_HPP_

// Blog: http://blog.csdn.net/fengbingchun/article/details/79346691

#include <vector>
#include <string>

namespace ANN {

template<typename T>
class LogisticRegression2 { // two categories
public:
	LogisticRegression2() = default;
	int init(const T* data, const T* labels, int train_num, int feature_length, T learning_rate = 0.00001, int iterations = 10000);
	int train(const std::string& model);
	int load_model(const std::string& model);
	T predict(const T* data, int feature_length) const; // y = 1/(1+exp(-(wx+b)))

private:
	int store_model(const std::string& model) const;
	T calculate_sigmoid(T value) const; // y = 1/(1+exp(-value))
	T calculate_z(const std::vector<T>& feature) const;

	std::vector<std::vector<T>> x; // training set
	std::vector<T> y; // ground truth labels
	int iterations = 1000;
	int m = 0; // train samples num
	int feature_length = 0;
	T alpha = (T)0.00001; // learning rate
	std::vector<T> w; // weights
	T b = (T)0.; // threshold
}; // class LogisticRegression2

} // namespace ANN

#endif // FBC_SRC_NN_LOGISTIC_REGRESSION2_HPP_
