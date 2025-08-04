#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <set>

#include "utils/csv_utils.h"
#include "utils/rolling_stats.h"
#include "algs/anomaly_sliding_window.h"
#include "algs/anomaly_heap.h"
#include <numeric>

void printDataAnalysis(const std::vector<double>& data) {
    if (data.empty()) return;
    
    auto minmax = std::minmax_element(data.begin(), data.end());
    double min_val = *minmax.first;
    double max_val = *minmax.second;
    
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    double mean = sum / data.size();
    
    // Calculate standard deviation
    double sq_sum = 0.0;
    for (double val : data) {
        sq_sum += (val - mean) * (val - mean);
    }
    double std_dev = std::sqrt(sq_sum / data.size());
    
    // Count different change magnitudes
    int zero_count = 0;
    int small_count = 0;  // < 1%
    int medium_count = 0; // 1-3%
    int large_count = 0;  // 3-5%
    int extreme_count = 0; // > 5%
    
    for (double val : data) {
        double abs_val = std::abs(val);
        if (abs_val < 1e-10) zero_count++;
        else if (abs_val < 0.01) small_count++;
        else if (abs_val < 0.03) medium_count++;
        else if (abs_val < 0.05) large_count++;
        else extreme_count++;
    }
    
    std::cout << "=== DATA ANALYSIS ===" << std::endl;
    std::cout << "Total data points: " << data.size() << std::endl;
    std::cout << "Mean: " << std::fixed << std::setprecision(6) << mean << std::endl;
    std::cout << "Standard deviation: " << std_dev << std::endl;
    std::cout << "Min value: " << min_val << std::endl;
    std::cout << "Max value: " << max_val << std::endl;
    std::cout << "Range: " << (max_val - min_val) << std::endl;
    std::cout << "Zero values: " << zero_count << std::endl;
    std::cout << "Small changes (<1%): " << small_count << std::endl;
    std::cout << "Medium changes (1-3%): " << medium_count << std::endl;
    std::cout << "Large changes (3-5%): " << large_count << std::endl;
    std::cout << "Extreme changes (>5%): " << extreme_count << std::endl;
    std::cout << "===================" << std::endl << std::endl;
}

void saveAnomalies(const std::vector<int>& anomalies, const std::string& filename, 
                   const std::string& method) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for writing" << std::endl;
        return;
    }
    
    file << "index,method" << std::endl;
    for (int idx : anomalies) {
        file << idx << "," << method << std::endl;
    }
    file.close();
    std::cout << "• " << filename << std::endl;
}

void printSummary(const std::vector<double>& data, 
                  const std::vector<int>& sliding_anomalies,
                  const std::vector<int>& heap_anomalies) {
    
    // Calculate overlap
    std::set<int> sliding_set(sliding_anomalies.begin(), sliding_anomalies.end());
    std::set<int> heap_set(heap_anomalies.begin(), heap_anomalies.end());
    
    std::vector<int> overlap;
    std::set_intersection(sliding_set.begin(), sliding_set.end(),
                         heap_set.begin(), heap_set.end(),
                         std::back_inserter(overlap));
    
    double sliding_pct = (double)sliding_anomalies.size() / data.size() * 100;
    double heap_pct = (double)heap_anomalies.size() / data.size() * 100;
    double overlap_pct = (double)overlap.size() / data.size() * 100;
    
    std::cout << "===================================================" << std::endl;
    std::cout << "FINAL RESULTS SUMMARY" << std::endl;
    std::cout << "===================================================" << std::endl;
    std::cout << "🔢 Total data points analyzed: " << data.size() << std::endl;
    std::cout << "🔍 Sliding window anomalies: " << sliding_anomalies.size() 
              << " (" << std::fixed << std::setprecision(5) << sliding_pct << "%)" << std::endl;
    std::cout << "🔍 Heap-based anomalies: " << heap_anomalies.size() 
              << " (" << heap_pct << "%)" << std::endl;
    std::cout << "🔄 Overlapping anomalies: " << overlap.size() 
              << " (" << overlap_pct << "%)" << std::endl;
    
    std::cout << "🔍 DETECTION QUALITY ASSESSMENT:" << std::endl;
    if (sliding_pct >= 2.0 && sliding_pct <= 5.0) {
        std::cout << "✅ Sliding window: Good detection rate" << std::endl;
    } else if (sliding_pct < 2.0) {
        std::cout << "⚠️ Sliding window: Low detection rate (may miss anomalies)" << std::endl;
    } else {
        std::cout << "⚠️ Sliding window: High detection rate (may be oversensitive)" << std::endl;
    }
    
    if (heap_pct >= 2.0 && heap_pct <= 6.0) {
        std::cout << "✅ Heap-based: Good detection rate" << std::endl;
    } else if (heap_pct < 2.0) {
        std::cout << "⚠️ Heap-based: Low detection rate (may miss anomalies)" << std::endl;
    } else {
        std::cout << "⚠️ Heap-based: High detection rate (may be oversensitive)" << std::endl;
    }
    
    std::cout << "🔄 Results written to:" << std::endl;
}

int main() {
    // Load data
    std::string filename = "../data/features.csv";
    std::vector<double> data;
    
    std::cout << "Loading data from " << filename << "..." << std::endl;
    
    // Try to load the CSV file
    if (!loadCSV(filename, data)) {
        std::cerr << "Failed to load data from " << filename << std::endl;
        return 1;
    }
    
    std::cout << "Loaded " << data.size() << " rows from " << filename << std::endl << std::endl;
    
    // Print data analysis
    printDataAnalysis(data);
    
    // === SLIDING WINDOW DETECTION ===
    std::cout << "=== SLIDING WINDOW DETECTION ===" << std::endl;
    int window_size = 30;
    double threshold_std = 2.5;
    
    std::cout << "Window size: " << window_size << std::endl;
    std::cout << "Threshold: " << threshold_std << " standard deviations" << std::endl;
    
    auto sliding_anomalies = detectAnomaliesSlidingWindow(data, window_size, threshold_std);
    
    double sliding_percentage = (double)sliding_anomalies.size() / data.size() * 100;
    std::cout << "✅ Sliding window anomalies detected: " << sliding_anomalies.size() 
              << " (" << std::fixed << std::setprecision(5) << sliding_percentage << "%)" << std::endl << std::endl;
    
    // === IMPROVED HEAP-BASED DETECTION ===
    std::cout << "=== IMPROVED HEAP-BASED DETECTION ===" << std::endl;
    
    // Use the granular search with a target of ~3-4% (similar to sliding window)
    double target_rate = 0.035; // 3.5% target
    auto heap_anomalies = detectAnomaliesHeapGranular(data, target_rate);
    
    double heap_percentage = (double)heap_anomalies.size() / data.size() * 100;
    std::cout << "Final heap detection rate: " << heap_percentage << "%" << std::endl << std::endl;
    
    // Print final summary
    printSummary(data, sliding_anomalies, heap_anomalies);
    
    // Save results
    saveAnomalies(sliding_anomalies, "../output/sliding_anomalies.csv", "sliding_window");
    saveAnomalies(heap_anomalies, "../output/heap_anomalies.csv", "heap_based");
    
    std::cout << "===================================================" << std::endl;
    
    std::cout << std::endl << "🚀 Run 'python improved_anomaly_visualization.py' for enhanced visualizations!" << std::endl;
    
    return 0;
}

    