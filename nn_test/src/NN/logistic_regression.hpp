#ifndef FBC_NN_LOGISTICREGRESSION_HPP_
#define FBC_NN_LOGISTICREGRESSION_HPP_

// Blog: http://blog.csdn.net/fengbingchun/article/details/78283675

#include <string>
#include <memory>
#include <vector>

namespace ANN {

template<typename T>
class LogisticRegression { // two categories
public:
	LogisticRegression() = default;
	int init(const T* data, const T* labels, int train_num, int feature_length,
		int reg_kinds = -1, T learning_rate = 0.00001, int iterations = 10000, int train_method = 0, int mini_batch_size = 1);
	int train(const std::string& model);
	int load_model(const std::string& model);
	T predict(const T* data, int feature_length) const; // y = 1/(1+exp(-(wx+b)))

	// Regularization kinds
	enum RegKinds {
		REG_DISABLE = -1, // Regularization disabled
		REG_L1 = 0 // L1 norm
	};

	// Training methods
	enum Methods {
		BATCH = 0,
		MINI_BATCH = 1
	};

private:
	int store_model(const std::string& model) const;
	T calc_sigmoid(T x) const; // y = 1/(1+exp(-x))
	T norm(const std::vector<T>& v1, const std::vector<T>& v2) const;
	void batch_gradient_descent();
	void mini_batch_gradient_descent();
	void gradient_descent(const std::vector<std::vector<T>>& data_batch, const std::vector<T>& labels_batch, int length_batch);

	std::vector<std::vector<T>> data;
	std::vector<T> labels;
	int iterations = 1000;
	int train_num = 0; // train samples num
	int feature_length = 0;
	T learning_rate = 0.00001;
	std::vector<T> thetas; // coefficient
	//T epsilon = 0.000001; // termination condition
	T lambda = (T)0.; // regularization method
	int train_method = 0;
	int mini_batch_size = 1;
};

} // namespace ANN

#endif // FBC_NN_LOGISTICREGRESSION_HPP_
