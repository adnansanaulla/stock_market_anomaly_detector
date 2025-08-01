#include <cmath>
#include <random>

#include "../../include/common.hpp"

vector<int> kmeans(const vector<vector<double>>& data, int k, int max_iters) {
    int n = data.size(), dims = data[0].size();
    vector<vector<double>> centroids(k, vector<double>(dims));
    vector<int> labels(n);

    // Initialize centroids randomly
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, n - 1);
    for (int i = 0; i < k; ++i) centroids[i] = data[dis(gen)];

    for (int iter = 0; iter < max_iters; ++iter) {
        // Assign clusters
        for (int i = 0; i < n; ++i) {
            double min_dist = 1e9;
            for (int j = 0; j < k; ++j) {
                double dist = euclidean_distance(data[i], centroids[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    labels[i] = j;
                }
            }
        }

        // Update centroids
        vector<vector<double>> new_centroids(k, vector<double>(dims, 0.0));
        vector<int> counts(k, 0);
        for (int i = 0; i < n; ++i) {
            counts[labels[i]]++;
            for (int d = 0; d < dims; ++d)
                new_centroids[labels[i]][d] += data[i][d];
        }
        for (int j = 0; j < k; ++j)
            if (counts[j] != 0)
                for (int d = 0; d < dims; ++d)
                    centroids[j][d] = new_centroids[j][d] / counts[j];
    }
    return labels;
}
