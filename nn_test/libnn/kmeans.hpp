#ifndef FBC_SRC_NN_KMEANS_HPP_
#define FBC_SRC_NN_KMEANS_HPP_

// Blog: http://blog.csdn.net/fengbingchun/article/details/79511318

#include <vector>

namespace ANN {

typedef enum KmeansFlags {
	// Select random initial centers in each attempt
	KMEANS_RANDOM_CENTERS = 0,
	// Use kmeans++ center initialization by Arthur and Vassilvitskii [Arthur2007]
	//KMEANS_PP_CENTERS = 2,
	// During the first (and possibly the only) attempt, use the
	//user-supplied labels instead of computing them from the initial centers. For the second and
	//further attempts, use the random or semi-random centers. Use one of KMEANS_\*_CENTERS flag
	//to specify the exact method.
	//KMEANS_USE_INITIAL_LABELS = 1
} KmeansFlags;

template<typename T>
int kmeans(const std::vector<std::vector<T>>& data, int K, std::vector<int>& best_labels, std::vector<std::vector<T>>& centers, double& compactness_measure,
	int max_iter_count = 100, double epsilon = 0.001, int attempts = 3, int flags = KMEANS_RANDOM_CENTERS);

} // namespace ANN

#endif // FBC_SRC_NN_KMEANS_HPP_

