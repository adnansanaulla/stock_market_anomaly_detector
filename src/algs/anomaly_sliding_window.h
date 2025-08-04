#pragma once
#include <vector>

std::vector<int> detectAnomaliesSlidingWindow(const std::vector<double>& series, 
                                             int window_size, 
                                             double threshold);