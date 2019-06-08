#include "logistic_regression2.hpp"
#include <fstream>
#include <algorithm>
#include <random>
#include <cmath>
#include "common.hpp"

namespace ANN {

template<typename T>
int LogisticRegression2<T>::init(const T* data, const T* labels, int train_num, int feature_length, T learning_rate, int iterations)
{
	if (train_num < 2) {
		fprintf(stderr, "logistic regression train samples num is too little: %d\n", train_num);
		return -1;
	}
	if (learning_rate <= 0) {
		fprintf(stderr, "learning rate must be greater 0: %f\n", learning_rate);
		return -1;
	}
	if (iterations <= 0) {
		fprintf(stderr, "number of iterations cannot be zero or a negative number: %d\n", iterations);
		return -1;
	}

	this->alpha = learning_rate;
	this->iterations = iterations;

	this->m = train_num;
	this->feature_length = feature_length;

	this->x.resize(train_num);
	this->y.resize(train_num);

	for (int i = 0; i < train_num; ++i) {
		const T* p = data + i * feature_length;
		this->x[i].resize(feature_length);

		for (int j = 0; j < feature_length; ++j) {
			this->x[i][j] = p[j];
		}

		this->y[i] = labels[i];
	}

	return 0;
}

template<typename T>
T LogisticRegression2<T>::calculate_z(const std::vector<T>& feature) const
{
	T z{ 0. };
	for (int i = 0; i < this->feature_length; ++i) {
		z += w[i] * feature[i];
	}
	z += b;

	return z;
}

template<typename T>
int LogisticRegression2<T>::train(const std::string& model)
{
	CHECK(x.size() == y.size());

	w.resize(this->feature_length, (T)0.);
	std::random_device rd;
	std::mt19937 generator(rd());
	std::uniform_real_distribution<T> distribution(-0.01, 0.01);
	for (int i = 0; i < this->feature_length; ++i) {
		w[i] = distribution(generator);
	}
	b = distribution(generator);

	for (int iter = 0; iter < this->iterations; ++iter) {
		T J = (T)0., db = (T)0.;
		std::vector<T> dw(this->feature_length, (T)0.);
		std::vector<T> z(this->m, (T)0), a(this->m, (T)0), dz(this->m, (T)0);

		for (int i = 0; i < this->m; ++i) {
			z[i] = calculate_z(x[i]); // z(i)=w^T*x(i)+b
			a[i] = calculate_sigmoid(z[i]); // a(i)= 1/(1+e^(-z(i)))
			J += -(y[i] * std::log(a[i]) + (1 - y[i] * std::log(1 - a[i]))); // J+=-[y(i)*loga(i)+(1-y(i))*log(1-a(i))]
			dz[i] = a[i] - y[i]; // dz(i) = a(i)-y(i)

			for (int j = 0; j < this->feature_length; ++j) {
				dw[j] += x[i][j] * dz[i]; // dw(i)+=x(i)(j)*dz(i)
			}
			db += dz[i]; // db+=dz(i)
		}

		J /= this->m;
		for (int j = 0; j < this->feature_length; ++j) {
			dw[j] /= m;
		}
		db /= m;

		for (int j = 0; j < this->feature_length; ++j) {
			w[j] -= this->alpha * dw[j];
		}
		b -= this->alpha*db;
	}

	CHECK(store_model(model) == 0);

	return 0;
}

template<typename T>
int LogisticRegression2<T>::load_model(const std::string& model)
{
	std::ifstream file;
	file.open(model.c_str(), std::ios::binary);
	if (!file.is_open()) {
		fprintf(stderr, "open file fail: %s\n", model.c_str());
		return -1;
	}

	int length{ 0 };
	file.read((char*)&length, sizeof(length));
	this->w.resize(length);
	this->feature_length = length;
	file.read((char*)this->w.data(), sizeof(T)*this->w.size());
	file.read((char*)&this->b, sizeof(T));

	file.close();

	return 0;
}

template<typename T>
T LogisticRegression2<T>::predict(const T* data, int feature_length) const
{
	CHECK(feature_length == this->feature_length);

	T value{ (T)0. };
	for (int t = 0; t < this->feature_length; ++t) {
		value += data[t] * this->w[t];
	}
	value += this->b;

	return (calculate_sigmoid(value));
}

template<typename T>
int LogisticRegression2<T>::store_model(const std::string& model) const
{
	std::ofstream file;
	file.open(model.c_str(), std::ios::binary);
	if (!file.is_open()) {
		fprintf(stderr, "open file fail: %s\n", model.c_str());
		return -1;
	}

	int length = w.size();
	file.write((char*)&length, sizeof(length));
	file.write((char*)w.data(), sizeof(T) * w.size());
	file.write((char*)&b, sizeof(T));

	file.close();

	return 0;
}

template<typename T>
T LogisticRegression2<T>::calculate_sigmoid(T value) const
{
	return ((T)1 / ((T)1 + exp(-value)));
}

template class LogisticRegression2<float>;
template class LogisticRegression2<double>;

} // namespace ANN

