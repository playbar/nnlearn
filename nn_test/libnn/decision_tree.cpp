#include "decision_tree.hpp"
#include <set>
#include <algorithm>
#include <typeinfo>
#include <iterator>
#include "common.hpp"

namespace ANN {

template<typename T>
int DecisionTree<T>::init(const std::vector<std::vector<T>>& data, const std::vector<T>& classes)
{
	CHECK(data.size() != 0 && classes.size() != 0 && data[0].size() != 0);

	this->samples_num = data.size();
	this->classes_num = classes.size();
	this->feature_length = data[0].size() -1;

	for (int i = 0; i < this->samples_num; ++i) {
		this->src_data.emplace_back(data[i]);
	}

	return 0;
}

template<typename T>
T DecisionTree<T>::gini_index(const std::vector<std::vector<std::vector<T>>>& groups, const std::vector<T>& classes) const
{
	// Gini calculation for a group
	// proportion = count(class_value) / count(rows)
	// gini_index = (1.0 - sum(proportion * proportion)) * (group_size/total_samples)

	// count all samples at split point
	int instances = 0;
	int group_num = groups.size();
	for (int i = 0; i < group_num; ++i) {
		instances += groups[i].size();
	}

	// sum weighted Gini index for each group
	T gini = (T)0.;
	for (int i = 0; i < group_num; ++i) {
		int size = groups[i].size();
		// avoid divide by zero
		if (size == 0) continue;
		T score = (T)0.;

		// score the group based on the score for each class
		T p = (T)0.;
		for (int c = 0; c < classes.size(); ++c) {
			int count = 0;
			for (int t = 0; t < size; ++t) {
				if (groups[i][t][this->feature_length] == classes[c]) ++count;
			}
			T p = (float)count / size;
			score += p * p;
		}

		// weight the group score by its relative size
		gini += (1. - score) * (float)size / instances;
	}

	return gini;
}

template<typename T>
std::vector<std::vector<std::vector<T>>> DecisionTree<T>::test_split(int index, T value, const std::vector<std::vector<T>>& dataset) const
{
	std::vector<std::vector<std::vector<T>>> groups(2); // 0: left, 1: reight

	for (int row = 0; row < dataset.size(); ++row) {
		if (dataset[row][index] < value) {
			groups[0].emplace_back(dataset[row]);
		} else {
			groups[1].emplace_back(dataset[row]);
		}
	}

	return groups;
}

template<typename T>
std::tuple<int, T, std::vector<std::vector<std::vector<T>>>> DecisionTree<T>::get_split(const std::vector<std::vector<T>>& dataset) const
{
	std::vector<T> values;
	for (int i = 0; i < dataset.size(); ++i) {
		values.emplace_back(dataset[i][this->feature_length]);
	}

	std::set<T> vals(values.cbegin(), values.cend());
	std::vector<T> class_values(vals.cbegin(), vals.cend());

	int b_index = 999;
	T b_value = (T)999.;
	T b_score = (T)999.;
	std::vector<std::vector<std::vector<T>>> b_groups(2);

	for (int index = 0; index < this->feature_length; ++index) {
		for (int row = 0; row < dataset.size(); ++row) {
			std::vector<std::vector<std::vector<T>>> groups = test_split(index, dataset[row][index], dataset);
			T gini = gini_index(groups, class_values);

			if (gini < b_score) {
				b_index = index;
				b_value = dataset[row][index];
				b_score = gini;
				b_groups = groups;
			}
		}
	}

	// a new node: the index of the chosen attribute, the value of that attribute by which to split and the two groups of data split by the chosen split point
	return std::make_tuple(b_index, b_value, b_groups);
}

template<typename T>
T DecisionTree<T>::to_terminal(const std::vector<std::vector<T>>& group) const
{
	std::vector<T> values;
	for (int i = 0; i < group.size(); ++i) {
		values.emplace_back(group[i][this->feature_length]);
	}

	std::set<T> vals(values.cbegin(), values.cend());
	int max_count = -1, index = -1;
	for (int i = 0; i < vals.size(); ++i) {
		int count = std::count(values.cbegin(), values.cend(), *std::next(vals.cbegin(), i));
		if (max_count < count) {
			max_count = count;
			index = i;
		}
	}

	return *std::next(vals.cbegin(), index);
}

template<typename T>
void DecisionTree<T>::split(binary_tree* node, int depth)
{
	std::vector<std::vector<T>> left = std::get<2>(node->dict)[0];
	std::vector<std::vector<T>> right = std::get<2>(node->dict)[1];
	std::get<2>(node->dict).clear();

	// check for a no split
	if (left.size() == 0 || right.size() == 0) {
		for (int i = 0; i < right.size(); ++i) {
			left.emplace_back(right[i]);
		}

		node->class_value_left = node->class_value_right = to_terminal(left);
		return;
	}

	// check for max depth
	if (depth >= max_depth) {
		node->class_value_left = to_terminal(left);
		node->class_value_right = to_terminal(right);
		return;
	}

	// process left child
	if (left.size() <= min_size) {
		node->class_value_left = to_terminal(left);
	} else {
		dictionary dict = get_split(left);
		node->left = new binary_tree;
		node->left->dict = dict;
		split(node->left, depth+1);
	}

	// process right child
	if (right.size() <= min_size) {
		node->class_value_right = to_terminal(right);
	} else {
		dictionary dict = get_split(right);
		node->right = new binary_tree;
		node->right->dict = dict;
		split(node->right, depth+1);
	}
}

template<typename T>
void DecisionTree<T>::build_tree(const std::vector<std::vector<T>>& train)
{
	// create root node
	dictionary root = get_split(train);
	binary_tree* node = new binary_tree;
	node->dict = root;
	tree = node;
	split(node, 1);
}

template<typename T>
void DecisionTree<T>::train()
{
	this->max_nodes = (1 << max_depth) - 1;
	build_tree(src_data);

	accuracy_metric();
	
	//binary_tree* tmp = tree;
	//print_tree(tmp);
}

template<typename T>
T DecisionTree<T>::predict(const std::vector<T>& data) const
{
	if (!tree) {
		fprintf(stderr, "Error, tree is null\n");
		return -1111.f;
	}

	return predict(tree, data);
}

template<typename T>
T DecisionTree<T>::predict(binary_tree* node, const std::vector<T>& data) const
{
	if (data[std::get<0>(node->dict)] < std::get<1>(node->dict)) {
		if (node->left) {
			return predict(node->left, data);
		} else {
			return node->class_value_left;
		}
	} else {
		if (node->right) {
			return predict(node->right, data);
		} else {
			return node->class_value_right;
		}
	}
}

template<typename T>
int DecisionTree<T>::save_model(const char* name) const
{
	std::ofstream file(name, std::ios::out);
	if (!file.is_open()) {
		fprintf(stderr, "open file fail: %s\n", name);
		return -1;
	}

	file<<max_depth<<","<<min_size<<std::endl;
	
	binary_tree* tmp = tree;
	int depth = height_of_tree(tmp);
	CHECK(max_depth == depth);
	
	tmp = tree;
	write_node(tmp, file);

	file.close();
	return 0;
}

template<typename T>
void DecisionTree<T>::write_node(const binary_tree* node, std::ofstream& file) const
{
	/*if (!node) return;

	write_node(node->left, file);
	file<<std::get<0>(node->dict)<<","<<std::get<1>(node->dict)<<","<<node->class_value_left<<","<<node->class_value_right<<std::endl;
	write_node(node->right, file);*/
	
	//typedef std::tuple<int, int, T, T, T> row; // flag, index, value, class_value_left, class_value_right
	std::vector<row_element> vec(this->max_nodes, std::make_tuple(-1, -1, (T)-1.f, (T)-1.f, (T)-1.f));

	binary_tree* tmp = const_cast<binary_tree*>(node);
	node_to_row_element(tmp, vec, 0);

	for (const auto& row : vec) {
		file<<std::get<0>(row)<<","<<std::get<1>(row)<<","<<std::get<2>(row)<<","<<std::get<3>(row)<<","<<std::get<4>(row)<<std::endl;
	}
}

template<typename T>
void DecisionTree<T>::node_to_row_element(binary_tree* node, std::vector<row_element>& rows, int pos) const
{
	if (!node) return;

	rows[pos] = std::make_tuple(0, std::get<0>(node->dict), std::get<1>(node->dict), node->class_value_left, node->class_value_right); // 0: have node, -1: no node
	
	if (node->left) node_to_row_element(node->left, rows, 2*pos+1);
	if (node->right) node_to_row_element(node->right, rows, 2*pos+2);
}

template<typename T>
int DecisionTree<T>::height_of_tree(const binary_tree* node) const
{
	if (!node)
		return 0;
	else
		return std::max(height_of_tree(node->left), height_of_tree(node->right)) + 1;
}

template<typename T>
int DecisionTree<T>::load_model(const char* name)
{
	std::ifstream file(name, std::ios::in);
	if (!file.is_open()) {
		fprintf(stderr, "open file fail: %s\n", name);
		return -1;
	}

	std::string line, cell;
	std::getline(file, line);
	std::stringstream line_stream(line);
	std::vector<int> vec;
	int count = 0;
	while (std::getline(line_stream, cell, ',')) {
		vec.emplace_back(std::stoi(cell));
	}
	CHECK(vec.size() == 2);
	max_depth = vec[0];
	min_size = vec[1];
	max_nodes = (1 << max_depth) - 1;
	std::vector<row_element> rows(max_nodes);
	
	if (typeid(float).name() == typeid(T).name()) {
		while (std::getline(file, line)) {
			std::stringstream line_stream2(line);
			std::vector<T> vec2;
		
			while(std::getline(line_stream2, cell, ',')) {
				vec2.emplace_back(std::stof(cell));
			}
			
			CHECK(vec2.size() == 5);
			rows[count] = std::make_tuple((int)vec2[0], (int)vec2[1], vec2[2], vec2[3], vec2[4]);
			//fprintf(stderr, "%d, %d, %f, %f, %f\n", std::get<0>(rows[count]), std::get<1>(rows[count]), std::get<2>(rows[count]), std::get<3>(rows[count]), std::get<4>(rows[count]));
			++count;
		}
	} else { // double
		while (std::getline(file, line)) {
			std::stringstream line_stream2(line);
			std::vector<T> vec2;
		
			while(std::getline(line_stream2, cell, ',')) {
				vec2.emplace_back(std::stod(cell));
			}

			CHECK(vec2.size() == 5);
			rows[count] = std::make_tuple((int)vec2[0], (int)vec2[1], vec2[2], vec2[3], vec[4]);
			++count;
		}
	}

	CHECK(max_nodes == count);
	CHECK(std::get<0>(rows[0]) != -1);

	binary_tree* tmp = new binary_tree;
	std::vector<std::vector<std::vector<T>>> dump;
	tmp->dict = std::make_tuple(std::get<1>(rows[0]), std::get<2>(rows[0]), dump);
	tmp->class_value_left = std::get<3>(rows[0]);
	tmp->class_value_right = std::get<4>(rows[0]);
	tree = tmp;
	row_element_to_node(tmp, rows, max_nodes, 0);

	file.close();
	return 0;
}

template<typename T>
void DecisionTree<T>::row_element_to_node(binary_tree* node, const std::vector<row_element>& rows, int n, int pos)
{
	if (!node || n == 0) return;

	int new_pos = 2 * pos + 1;
	if (new_pos < n && std::get<0>(rows[new_pos]) != -1) {
		node->left = new binary_tree;
		std::vector<std::vector<std::vector<T>>> dump;
		node->left->dict = std::make_tuple(std::get<1>(rows[new_pos]), std::get<2>(rows[new_pos]), dump);
		node->left->class_value_left = std::get<3>(rows[new_pos]);
		node->left->class_value_right = std::get<4>(rows[new_pos]);

		row_element_to_node(node->left, rows, n, new_pos);
	}

	new_pos = 2 * pos + 2;
	if (new_pos < n && std::get<0>(rows[new_pos]) != -1) {
		node->right = new binary_tree;
		std::vector<std::vector<std::vector<T>>> dump;
		node->right->dict = std::make_tuple(std::get<1>(rows[new_pos]), std::get<2>(rows[new_pos]), dump);
		node->right->class_value_left = std::get<3>(rows[new_pos]);
		node->right->class_value_right = std::get<4>(rows[new_pos]);
	
		row_element_to_node(node->right, rows, n, new_pos);
	}
}

template<typename T>
void DecisionTree<T>::delete_tree()
{
	delete_node(tree);
}

template<typename T>
void DecisionTree<T>::delete_node(binary_tree* node)
{
	if (node->left) delete_node(node->left);
	if (node->right) delete_node(node->right);
	delete node;
}

template<typename T>
double DecisionTree<T>::accuracy_metric() const
{
	int correct = 0;
	for (int i = 0; i < this->samples_num; ++i) {
		T predicted = predict(tree, src_data[i]);
		if (predicted == src_data[i][this->feature_length])
			++correct;
	}

	double accuracy = correct / (double)samples_num * 100.;
	fprintf(stdout, "train accuracy: %f\n", accuracy);

	return accuracy;  
}

template<typename T>
void DecisionTree<T>::print_tree(const binary_tree* node, int depth) const
{
	if (node) {
		std::string blank = " ";
		for (int i = 0; i < depth; ++i) blank += blank;
		fprintf(stdout, "%s[X%d < %.3f]\n", blank.c_str(), std::get<0>(node->dict)+1, std::get<1>(node->dict));

		if (!node->left || !node->right)
			blank += blank;

		if (!node->left)
			fprintf(stdout, "%s[%.1f]\n", blank.c_str(), node->class_value_left);
		else 
			print_tree(node->left, depth+1);

		if (!node->right)
			fprintf(stdout, "%s[%.1f]\n", blank.c_str(), node->class_value_right);
		else
			print_tree(node->right, depth+1);
			
	}
}

template class DecisionTree<float>;
template class DecisionTree<double>;

} // namespace ANN

