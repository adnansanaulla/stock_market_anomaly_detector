#include <vector>
#include <deque>
#include <algorithm>
#include <cmath>
#include <iostream>

std::vector<int> detectHeapAnomalies(const std::vector<double>& series, int window_size, double threshold) {
    std::vector<int> flags(series.size(), 0);
    std::deque<double> window;
    
    int anomaly_count = 0;
    double max_deviation = 0.0;
    
    for (size_t i = 0; i < series.size(); ++i) {
        // Maintain sliding window
        if (window.size() == window_size) {
            window.pop_front();
        }
        
        // Start detection after we have enough data
        if (window.size() >= window_size / 2) {
            // Simple but effective approach: compare to max/min in window
            double max_val = *std::max_element(window.begin(), window.end());
            double min_val = *std::min_element(window.begin(), window.end());
            
            // Calculate simple statistics
            double sum = 0.0;
            for (double val : window) {
                sum += val;
            }
            double mean = sum / window.size();
            
            // Calculate standard deviation
            double variance = 0.0;
            for (double val : window) {
                variance += (val - mean) * (val - mean);
            }
            double stddev = std::sqrt(variance / window.size());
            
            bool is_anomaly = false;
            
            // Method 1: Distance from max/min (heap-like approach)
            double range = max_val - min_val;
            if (range > 0.001) {  // Only if there's meaningful variation
                double distance_from_max = std::abs(series[i] - max_val);
                double distance_from_min = std::abs(series[i] - min_val);
                double min_distance = std::min(distance_from_max, distance_from_min);
                
                // If current value extends the range significantly
                if (series[i] > max_val || series[i] < min_val) {
                    double extension = (series[i] > max_val) ? 
                                     (series[i] - max_val) : (min_val - series[i]);
                    
                    if (extension > threshold * range) {
                        is_anomaly = true;
                    }
                }
            }
            
            // Method 2: Standard deviation check (but less strict)
            if (!is_anomaly && stddev > 1e-6) {
                double z_score = std::abs(series[i] - mean) / stddev;
                if (z_score > threshold + 1.0) {  // Higher threshold for std dev
                    is_anomaly = true;
                }
            }
            
            // Method 3: Absolute value check for very large moves
            if (!is_anomaly && std::abs(series[i]) > 0.1) {  // 10% move is definitely notable
                is_anomaly = true;
            }
            
            if (is_anomaly) {
                flags[i] = 1;
                anomaly_count++;
                double deviation = (stddev > 1e-10) ? std::abs(series[i] - mean) / stddev : 0;
                max_deviation = std::max(max_deviation, deviation);
            }
        }
        
        window.push_back(series[i]);
    }
    
    std::cout << "Heap algorithm detected " << anomaly_count << " anomalies" << std::endl;
    std::cout << "Max deviation: " << max_deviation << std::endl;
    
    return flags;
}