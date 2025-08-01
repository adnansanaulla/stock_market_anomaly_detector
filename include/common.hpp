#ifndef COMMON_HPP
#define COMMON_HPP

#include <vector>
#include <string>
#include <cmath>

using namespace std;

vector<vector<double>> read_csv(const string& filename);
vector<int> kmeans(const vector<vector<double>>& data, int k, int max_iters);
vector<int> dbscan(const vector<vector<double>>& data, double eps, int min_pts);

inline double euclidean_distance(const vector<double>& a, const vector<double>& b) {
    double sum = 0;
    for (int i = 0; i < a.size(); ++i)
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    return sqrt(sum);
}

//test

#endif