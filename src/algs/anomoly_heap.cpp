#include <vector>
#include <deque>
#include <algorithm>
#include <cmath>

std::vector<int> detectHeapAnomalies(const std::vector<double>& series, int window_size, double threshold) {
    std::vector<int> flags(series.size(), 0);
    std::deque<double> window;

    for (size_t i = 0; i < series.size(); ++i) {
        double max_recent;

        if (window.size() == window_size) {
            window.pop_front();
        }

        if (window.empty()) {
            max_recent = 0.0;
        } else {
            max_recent = *std::max_element(window.begin(), window.end());
        }
        if (std::abs(series[i] - max_recent) > threshold) {
            flags[i] = 1;
        }
        window.push_back(series[i]);
    }

    return flags;
}
