#ifndef FBC_SRC_NN_SINGLE_HIDDEN_LAYER_HPP_
#define FBC_SRC_NN_SINGLE_HIDDEN_LAYER_HPP_

// Blog: http://blog.csdn.net/fengbingchun/article/details/79370310

#include <string>
#include <vector>

namespace ANN {

template<typename T>
class SingleHiddenLayer { // two categories
public:
	typedef enum ActivationFunctionType {
		Sigmoid = 0,
		TanH = 1,
		ReLU = 2,
		Leaky_ReLU = 3
	} ActivationFunctionType;

	SingleHiddenLayer() = default;
	int init(const T* data, const T* labels, int train_num, int feature_length,
		int hidden_layer_node_num = 20, T learning_rate = 0.00001, int iterations = 10000, int hidden_layer_activation_type = 2, int output_layer_activation_type = 0);
	int train(const std::string& model);
	int load_model(const std::string& model);
	T predict(const T* data, int feature_length) const;

private:
	T calculate_activation_function(T value, ActivationFunctionType type) const;
	T calcuate_activation_function_derivative(T value, ActivationFunctionType type) const;
	int store_model(const std::string& model) const;
	void init_train_variable();
	void init_w_and_b();

	ActivationFunctionType hidden_layer_activation_type = ReLU;
	ActivationFunctionType output_layer_activation_type = Sigmoid;
	std::vector<std::vector<T>> x; // training set
	std::vector<T> y; // ground truth labels
	int iterations = 10000;
	int m = 0; // train samples num
	int feature_length = 0;
	T alpha = (T)0.00001; // learning rate
	std::vector<std::vector<T>> w1, w2; // weights
	std::vector<T> b1, b2; // threshold
	int hidden_layer_node_num = 10;
	int output_layer_node_num = 1;
	T J = (T)0.;
	std::vector<std::vector<T>> dw1, dw2;
	std::vector<T> db1, db2;
	std::vector<std::vector<T>> z1, a1, z2, a2, da2, dz2, da1, dz1;
}; // class SingleHiddenLayer

} // namespace ANN

#endif // FBC_SRC_NN_SINGLE_HIDDEN_LAYER_HPP_
