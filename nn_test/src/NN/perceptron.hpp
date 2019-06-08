#ifndef _PERCEPTRON_HPP_
#define _PERCEPTRON_HPP_

// Blog: http://blog.csdn.net/fengbingchun/article/details/50097723

#include <vector>

namespace ANN {

typedef std::vector<float> feature;
typedef int label;

class Perceptron {
private:
	std::vector<feature> feature_set;
	std::vector<label> label_set;
	int iterates;
	float learn_rate;
	std::vector<float> weight;
	int size_weight;
	float bias;

	void initWeight();
	float calDotProduct(const feature feature_, const std::vector<float> weight_);
	void updateWeight(const feature feature_, int label_);

public:
	Perceptron(int iterates_, float learn_rate_, int size_weight_, float bias_);
	void getDataset(const std::vector<feature> feature_set_, const std::vector<label> label_set_);
	bool train();
	label predict(const feature feature_);
};

}


#endif // _PERCEPTRON_HPP_

