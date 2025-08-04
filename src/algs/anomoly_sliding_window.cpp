#include <vector>
#include <cmath>
#include "../utils/rolling_stats.h"

std::vector<int> detectSlidingAnomalies(const std::vector<double>& series, int window_size, double threshold) {
    std::vector<int> flags(series.size(), 0);
    RollingStats stats(window_size);

    for (size_t i = 0; i < series.size(); ++i) {
        stats.add(series[i]);

        if (stats.ready()) {
            double mean = stats.mean();
            double stddev = stats.stddev();

            if (std::abs(series[i] - mean) > threshold * stddev) {
                flags[i] = 1;
            }
        }
    }

    return flags;
}
