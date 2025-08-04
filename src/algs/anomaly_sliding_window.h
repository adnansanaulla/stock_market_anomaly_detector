#pragma once
#include <vector>

std::vector<int> detectSlidingAnomalies(const std::vector<double>& series, int window_size, double threshold);
