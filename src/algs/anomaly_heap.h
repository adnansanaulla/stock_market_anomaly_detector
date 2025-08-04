#pragma once
#include <vector>

std::vector<int> detectHeapAnomalies(const std::vector<double>& series, int window_size, double threshold);