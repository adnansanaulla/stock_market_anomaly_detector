#include "anomaly_heap.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

std::vector<int> detectAnomaliesHeap(const std::vector<double>& data, double threshold) {
    std::vector<int> anomalies;
    
    if (data.empty()) {
        return anomalies;
    }
    
    // Calculate robust statistics using median and MAD (Median Absolute Deviation)
    std::vector<double> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());
    
    // Calculate median
    double median;
    size_t n = sorted_data.size();
    if (n % 2 == 0) {
        median = (sorted_data[n/2 - 1] + sorted_data[n/2]) / 2.0;
    } else {
        median = sorted_data[n/2];
    }
    
    // Calculate MAD (Median Absolute Deviation)
    std::vector<double> deviations;
    deviations.reserve(data.size());
    for (double value : data) {
        deviations.push_back(std::abs(value - median));
    }
    std::sort(deviations.begin(), deviations.end());
    
    double mad;
    if (n % 2 == 0) {
        mad = (deviations[n/2 - 1] + deviations[n/2]) / 2.0;
    } else {
        mad = deviations[n/2];
    }
    
    // Convert MAD to standard deviation equivalent (MAD * 1.4826)
    double robust_std = mad * 1.4826;
    
    // Prevent division by zero
    if (robust_std < 1e-10) {
        robust_std = 1e-10;
    }
    
    // Use a more conservative threshold approach
    // Scale threshold based on data characteristics
    double adaptive_threshold = threshold;
    
    // For financial data, use a higher base threshold
    if (threshold < 2.0) {
        adaptive_threshold = std::max(threshold, 2.0);
    }
    
    // Create min-heap for tracking the most extreme deviations
    std::priority_queue<std::pair<double, int>, 
                       std::vector<std::pair<double, int>>, 
                       std::greater<std::pair<double, int>>> min_heap;
    
    // Calculate normalized deviations and identify potential anomalies
    std::vector<std::pair<double, int>> deviation_pairs;
    deviation_pairs.reserve(data.size());
    
    for (size_t i = 0; i < data.size(); ++i) {
        double normalized_deviation = std::abs(data[i] - median) / robust_std;
        deviation_pairs.push_back({normalized_deviation, static_cast<int>(i)});
    }
    
    // Sort by deviation (largest first)
    std::sort(deviation_pairs.begin(), deviation_pairs.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Select top anomalies more conservatively
    // Limit to a reasonable percentage of the data (similar to sliding window)
    size_t max_anomalies = static_cast<size_t>(data.size() * 0.05); // Max 5% of data
    size_t selected_anomalies = 0;
    
    // Use both threshold and ranking approach
    for (const auto& pair : deviation_pairs) {
        double deviation = pair.first;
        int index = pair.second;
        
        // Must exceed threshold AND be in top percentile
        if (deviation > adaptive_threshold && selected_anomalies < max_anomalies) {
            // Additional filter: ensure it's significantly different
            if (deviation > adaptive_threshold * 1.2) {  // 20% buffer above threshold
                anomalies.push_back(index);
                selected_anomalies++;
            }
        } else {
            break; // Stop if we've found enough or threshold not met
        }
    }
    
    // Sort anomaly indices
    std::sort(anomalies.begin(), anomalies.end());
    
    // Debug output
    std::cout << "Heap algorithm detected " << anomalies.size() << " anomalies" << std::endl;
    if (!deviation_pairs.empty()) {
        std::cout << "Max deviation: " << deviation_pairs[0].first << std::endl;
    }
    std::cout << "Median: " << median << ", Robust STD: " << robust_std << std::endl;
    std::cout << "Adaptive threshold: " << adaptive_threshold << std::endl;
    
    return anomalies;
}

// Alternative function with granular threshold search that's more conservative
std::vector<int> detectAnomaliesHeapGranular(const std::vector<double>& data,
                                            double target_percentage) {
    if (data.empty()) {
        return {};
    }
    
    // Try different thresholds but with a more focused range
    std::vector<double> thresholds;
    
    // Start with more reasonable thresholds for financial data
    for (double t = 3.5; t >= 2.0; t -= 0.1) {
        thresholds.push_back(t);
    }
    for (double t = 2.0; t >= 1.5; t -= 0.05) {
        thresholds.push_back(t);
    }
    
    double best_threshold = 3.5;
    std::vector<int> best_anomalies;
    double best_diff = std::numeric_limits<double>::max();
    
    std::cout << "=== HEAP-BASED DETECTION (GRANULAR SEARCH) ===" << std::endl;
    std::cout << "Target anomaly rate: " << (target_percentage * 100) << "%" << std::endl;
    std::cout << "Generated " << thresholds.size() << " threshold values to test" << std::endl;
    
    int attempt = 1;
    for (double threshold : thresholds) {
        std::cout << "[" << attempt << "/" << thresholds.size() << "] ";
        std::cout << "Trying threshold " << std::fixed << std::setprecision(6) << threshold << "..." << std::endl;
        
        auto anomalies = detectAnomaliesHeap(data, threshold);
        double percentage = static_cast<double>(anomalies.size()) / data.size();
        
        std::cout << "Found " << anomalies.size() << " anomalies (" 
                  << std::fixed << std::setprecision(5) << (percentage * 100) << "%)" << std::endl;
        
        // Check if this is closer to our target
        double diff = std::abs(percentage - target_percentage);
        if (diff < best_diff) {
            best_diff = diff;
            best_threshold = threshold;
            best_anomalies = anomalies;
        }
        
        // If we're close enough, stop searching
        if (percentage <= target_percentage * 1.2 && percentage >= target_percentage * 0.8) {
            std::cout << "✓ Good detection rate achieved!" << std::endl;
            break;
        } else if (percentage > target_percentage * 2) {
            std::cout << "⚠ Too many anomalies, trying higher threshold..." << std::endl;
        }
        
        attempt++;
    }
    
    std::cout << "🎯 Using best threshold found: " << best_threshold << std::endl;
    return best_anomalies;
}