#include "anomaly_sliding_window.h"
#include <vector>
#include <cmath>
#include "../utils/rolling_stats.h"

std::vector<int> detectAnomaliesSlidingWindow(const std::vector<double>& series, 
                                             int window_size, 
                                             double threshold) {
    std::vector<int> anomaly_indices;  
    RollingStats stats(window_size);

    for (size_t i = 0; i < series.size(); ++i) {
        stats.add(series[i]);

        if (stats.ready()) {
            double mean = stats.mean();
            double stddev = stats.stddev();

            if (std::abs(series[i] - mean) > threshold * stddev) {
                anomaly_indices.push_back(static_cast<int>(i));  // Store index
            }
        }
    }

    return anomaly_indices;
}