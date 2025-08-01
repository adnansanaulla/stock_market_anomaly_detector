#include "../include/common.hpp"
#include <fstream>

void write_clusters_to_csv(const string& filename, const vector<vector<double>>& data, const vector<int>& labels) {
    ofstream file(filename);
    file << "Return,HighLowSpread,VolumeZScore,Cluster\n";
    for (size_t i = 0; i < data.size(); ++i) {
        file << data[i][0] << "," << data[i][1] << "," << data[i][2] << "," << labels[i] << "\n";
    }
    file.close();
}

int main() {
    string path = "data/processed/features.csv";
    vector<vector<double>> data = read_csv(path);

    vector<int> kmeans_labels = kmeans(data, 3, 100);
    write_clusters_to_csv("output/kmeans_clusters.csv", data, kmeans_labels);

    vector<int> dbscan_labels = dbscan(data, 0.5, 5);
    write_clusters_to_csv("output/dbscan_clusters.csv", data, dbscan_labels);

    return 0;
}
