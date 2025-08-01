#include "../../include/common.hpp"

void dfs(int idx, int cluster_id, const vector<vector<double>>& data, vector<int>& labels, double eps, int min_pts) {
    labels[idx] = cluster_id;
    for (int i = 0; i < data.size(); ++i) {
        if (labels[i] == -1 && euclidean_distance(data[i], data[idx]) <= eps)
            dfs(i, cluster_id, data, labels, eps, min_pts);
    }
}

vector<int> dbscan(const vector<vector<double>>& data, double eps, int min_pts) {
    int n = data.size();
    vector<int> labels(n, -1);
    int cluster_id = 0;

    for (int i = 0; i < n; ++i) {
        if (labels[i] != -1) continue;
        int count = 0;
        for (int j = 0; j < n; ++j)
            if (euclidean_distance(data[i], data[j]) <= eps) count++;

        if (count >= min_pts)
            dfs(i, cluster_id++, data, labels, eps, min_pts);
    }
    return labels;
}