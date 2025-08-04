#include "rolling_stats.h"
#include <numeric>
#include <cmath>

RollingStats::RollingStats(int window_size) : window_size(window_size) {}

void RollingStats::add(double value) {
    if (window.size() == window_size) {
        window.pop_front();
    }
    window.push_back(value);
}

double RollingStats::mean() const {
    if (window.empty()) return 0.0;
    double sum = std::accumulate(window.begin(), window.end(), 0.0);
    return sum / window.size();
}

double RollingStats::stddev() const {
    if (window.empty()) return 0.0;
    double m = mean();
    double sq_sum = 0.0;
    for (double v : window) {
        sq_sum += (v - m) * (v - m);
    }
    return std::sqrt(sq_sum / window.size());
}

bool RollingStats::ready() const {
    return window.size() == window_size;
}
