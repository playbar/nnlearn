#include "perceptron.hpp"
#include <assert.h>
#include <time.h>
#include <iostream>

namespace ANN {

void Perceptron::updateWeight(const feature feature_, int label_)
{
	for (int i = 0; i < size_weight; i++) {
		weight[i] += learn_rate * feature_[i] * label_; // formula 5
	}

	bias += learn_rate * label_; // formula 5
}

float Perceptron::calDotProduct(const feature feature_, const std::vector<float> weight_)
{
	assert(feature_.size() == weight_.size());
	float ret = 0.;

	for (int i = 0; i < feature_.size(); i++) {
		ret += feature_[i] * weight_[i];
	}

	return ret;
}

void Perceptron::initWeight()
{
	srand(time(0));
	float range = 100.0;
	for (int i = 0; i < size_weight; i++) {
		float tmp = range * rand() / (RAND_MAX + 1.0);
		weight.push_back(tmp);
	}
}

Perceptron::Perceptron(int iterates_, float learn_rate_, int size_weight_, float bias_)
{
	iterates = iterates_;
	learn_rate = learn_rate_;
	size_weight = size_weight_;
	bias = bias_;
	weight.resize(0);
	feature_set.resize(0);
	label_set.resize(0);
}

void Perceptron::getDataset(const std::vector<feature> feature_set_, const std::vector<label> label_set_)
{
	assert(feature_set_.size() == label_set_.size());

	feature_set.resize(0);
	label_set.resize(0);

	for (int i = 0; i < feature_set_.size(); i++) {
		feature_set.push_back(feature_set_[i]);
		label_set.push_back(label_set_[i]);
	}
}

bool Perceptron::train()
{
	initWeight();

	for (int i = 0; i < iterates; i++) {
		bool flag = true;

		for (int j = 0; j < feature_set.size(); j++) {
			float tmp = calDotProduct(feature_set[j], weight) + bias;
			if (tmp * label_set[j] <= 0) {
				updateWeight(feature_set[j], label_set[j]);
				flag = false;
			}
		}

		if (flag) {
			std::cout << "iterate: " << i << std::endl;
			std::cout << "weight: ";
			for (int m = 0; m < size_weight; m++) {
				std::cout << weight[m] << "    ";
			}
			std::cout << std::endl;
			std::cout << "bias: " << bias << std::endl;

			return true;
		}
	}

	return false;
}

label Perceptron::predict(const feature feature_)
{
	assert(feature_.size() == size_weight);

	return calDotProduct(feature_, weight) + bias >= 0 ? 1 : -1; //formula 2
}

}