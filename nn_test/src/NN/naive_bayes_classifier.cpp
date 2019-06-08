#include "naive_bayes_classifier.hpp"
#include "common.hpp"
#include <math.h>
#include <iostream>
#include <algorithm>
#include <fstream>

namespace ANN {

template<typename T>
int NaiveBayesClassifier<T>::init(const std::vector<sex_info<T>>& info)
{
	int length = info.size();
	if (length < 2) {
		fprintf(stderr, "train data length should be > 1: %d\n", length);
		return -1;
	}

	male_train_number = 0;
	female_train_number = 0;

	for (int i = 0; i < length; ++i) {
		if (info[i].sex == 0) {
			++female_train_number;

			female_height.push_back(info[i].height);
			female_weight.push_back(info[i].weight);
			female_foot_size.push_back(info[i].foot_size);

		} else {
			++male_train_number;

			male_height.push_back(info[i].height);
			male_weight.push_back(info[i].weight);
			male_foot_size.push_back(info[i].foot_size);
		}
	}

	male_p = (T)male_train_number / (male_train_number + female_train_number);
	female_p = (T)female_train_number / (male_train_number + female_train_number);

	return 0;
}

template<typename T>
int NaiveBayesClassifier<T>::train(const std::string& model)
{
	std::tuple<T, T> mean_variance;

	calc_mean_variance(male_height, mean_variance);
	male_mv.mean_height = std::get<0>(mean_variance);
	male_mv.variance_height = std::get<1>(mean_variance);

	calc_mean_variance(male_weight, mean_variance);
	male_mv.mean_weight = std::get<0>(mean_variance);
	male_mv.variance_weight = std::get<1>(mean_variance);

	calc_mean_variance(male_foot_size, mean_variance);
	male_mv.mean_foot_size = std::get<0>(mean_variance);
	male_mv.variance_foot_size = std::get<1>(mean_variance);

	calc_mean_variance(female_height, mean_variance);
	female_mv.mean_height = std::get<0>(mean_variance);
	female_mv.variance_height = std::get<1>(mean_variance);

	calc_mean_variance(female_weight, mean_variance);
	female_mv.mean_weight = std::get<0>(mean_variance);
	female_mv.variance_weight = std::get<1>(mean_variance);

	calc_mean_variance(female_foot_size, mean_variance);
	female_mv.mean_foot_size = std::get<0>(mean_variance);
	female_mv.variance_foot_size = std::get<1>(mean_variance);

	CHECK(store_model(model) == 0);

	return 0;
}

template<typename T>
int NaiveBayesClassifier<T>::store_model(const std::string& model) const
{
	std::ofstream file;
	file.open(model.c_str(), std::ios::binary);
	if (!file.is_open()) {
		fprintf(stderr, "open file fail: %s\n", model.c_str());
		return -1;
	}

	file.write((char*)&male_p, sizeof(male_p));
	file.write((char*)&male_mv.mean_height, sizeof(male_mv.mean_height));
	file.write((char*)&male_mv.mean_weight, sizeof(male_mv.mean_weight));
	file.write((char*)&male_mv.mean_foot_size, sizeof(male_mv.mean_foot_size));
	file.write((char*)&male_mv.variance_height, sizeof(male_mv.variance_height));
	file.write((char*)&male_mv.variance_weight, sizeof(male_mv.variance_weight));
	file.write((char*)&male_mv.variance_foot_size, sizeof(male_mv.variance_foot_size));

	file.write((char*)&female_p, sizeof(female_p));
	file.write((char*)&female_mv.mean_height, sizeof(female_mv.mean_height));
	file.write((char*)&female_mv.mean_weight, sizeof(female_mv.mean_weight));
	file.write((char*)&female_mv.mean_foot_size, sizeof(female_mv.mean_foot_size));
	file.write((char*)&female_mv.variance_height, sizeof(female_mv.variance_height));
	file.write((char*)&female_mv.variance_weight, sizeof(female_mv.variance_weight));
	file.write((char*)&female_mv.variance_foot_size, sizeof(female_mv.variance_foot_size));

	file.close();

	return 0;
}

template<typename T>
int NaiveBayesClassifier<T>::predict(const sex_info<T>& info) const
{
	T male_height_p = calc_attribute_probability(info.height, male_mv.mean_height, male_mv.variance_height);
	T male_weight_p = calc_attribute_probability(info.weight, male_mv.mean_weight, male_mv.variance_weight);
	T male_foot_size_p = calc_attribute_probability(info.foot_size, male_mv.mean_foot_size, male_mv.variance_foot_size);

	T female_height_p = calc_attribute_probability(info.height, female_mv.mean_height, female_mv.variance_height);
	T female_weight_p = calc_attribute_probability(info.weight, female_mv.mean_weight, female_mv.variance_weight);
	T female_foot_size_p = calc_attribute_probability(info.foot_size, female_mv.mean_foot_size, female_mv.variance_foot_size);

	T evidence = male_p * male_height_p * male_weight_p * male_foot_size_p +
		female_p * female_height_p * female_weight_p * female_foot_size_p;

	T male_posterior = male_p * male_height_p * male_weight_p * male_foot_size_p /*/ evidence*/;
	T female_posterior = female_p * female_height_p * female_weight_p * female_foot_size_p /*/ evidence*/;

	fprintf(stdout, "male posterior probability: %e, female posterior probability: %e\n",
		male_posterior, female_posterior);

	if (male_posterior > female_posterior) return 1;
	else return 0;
}

template<typename T>
T NaiveBayesClassifier<T>::calc_attribute_probability(T value, T mean, T variance) const
{
	return (T)1 / std::sqrt(2 * PI * variance) * std::exp(-std::pow(value - mean, 2) / (2 * variance));
}

template<typename T>
int NaiveBayesClassifier<T>::load_model(const std::string& model) const
{
	std::ifstream file;
	file.open(model.c_str(), std::ios::binary);
	if (!file.is_open()) {
		fprintf(stderr, "open file fail: %s\n", model.c_str());
		return -1;
	}

	file.read((char*)&male_p, sizeof(male_p) * 1);
	file.read((char*)&male_mv.mean_height, sizeof(male_mv.mean_height) * 1);
	file.read((char*)&male_mv.mean_weight, sizeof(male_mv.mean_weight) * 1);
	file.read((char*)&male_mv.mean_foot_size, sizeof(male_mv.mean_foot_size) * 1);
	file.read((char*)&male_mv.variance_height, sizeof(male_mv.variance_height) * 1);
	file.read((char*)&male_mv.variance_weight, sizeof(male_mv.variance_weight) * 1);
	file.read((char*)&male_mv.variance_foot_size, sizeof(male_mv.variance_foot_size) * 1);

	file.read((char*)&female_p, sizeof(female_p)* 1);
	file.read((char*)&female_mv.mean_height, sizeof(female_mv.mean_height) * 1);
	file.read((char*)&female_mv.mean_weight, sizeof(female_mv.mean_weight) * 1);
	file.read((char*)&female_mv.mean_foot_size, sizeof(female_mv.mean_foot_size) * 1);
	file.read((char*)&female_mv.variance_height, sizeof(female_mv.variance_height) * 1);
	file.read((char*)&female_mv.variance_weight, sizeof(female_mv.variance_weight) * 1);
	file.read((char*)&female_mv.variance_foot_size, sizeof(female_mv.variance_foot_size) * 1);

	file.close();

	return 0;
}

template<typename T>
void NaiveBayesClassifier<T>::calc_mean_variance(const std::vector<T>& data, std::tuple<T, T>& mean_variance) const
{
	T sum{ 0 }, sqsum{ 0 };

	for (int i = 0; i < data.size(); ++i) {
		sum += data[i];
	}

	T mean = sum / data.size();

	for (int i = 0; i < data.size(); ++i) {
		sqsum += std::pow(data[i] - mean, 2);
	}

	// unbiased sample variances
	T variance = sqsum / (data.size() - 1);

	std::get<0>(mean_variance) = mean;
	std::get<1>(mean_variance) = variance;
}

template class NaiveBayesClassifier<float>;
template class NaiveBayesClassifier<double>;

} // namespace ANN
