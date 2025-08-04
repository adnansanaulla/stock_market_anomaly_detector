#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <iomanip>

#include "utils/csv_utils.h"
#include "algs/anomaly_sliding_window.h"
#include "algs/anomaly_heap.h"

void analyzeData(const std::vector<double>& series) {
    if (series.empty()) return;
    
    double sum = 0.0, sum_sq = 0.0;
    double min_val = series[0], max_val = series[0];
    
    for (double val : series) {
        sum += val;
        sum_sq += val * val;
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
    }
    
    double mean = sum / series.size();
    double variance = (sum_sq / series.size()) - (mean * mean);
    double stddev = std::sqrt(variance);
    
    std::cout << "=== DATA ANALYSIS ===" << std::endl;
    std::cout << "Total data points: " << series.size() << std::endl;
    std::cout << "Mean: " << mean << std::endl;
    std::cout << "Standard deviation: " << stddev << std::endl;
    std::cout << "Min value: " << min_val << std::endl;
    std::cout << "Max value: " << max_val << std::endl;
    std::cout << "Range: " << (max_val - min_val) << std::endl;
    
    // Count significant values
    int zero_count = 0;
    int small_changes = 0;
    int medium_changes = 0;
    int large_changes = 0;
    int extreme_changes = 0;
    
    for (double val : series) {
        double abs_val = std::abs(val);
        if (abs_val < 1e-10) zero_count++;
        else if (abs_val < 0.01) small_changes++;      // <1%
        else if (abs_val < 0.03) medium_changes++;     // 1-3%
        else if (abs_val < 0.05) large_changes++;      // 3-5%
        else extreme_changes++;                        // >5%
    }
    
    std::cout << "Zero values: " << zero_count << std::endl;
    std::cout << "Small changes (<1%): " << small_changes << std::endl;
    std::cout << "Medium changes (1-3%): " << medium_changes << std::endl;
    std::cout << "Large changes (3-5%): " << large_changes << std::endl;
    std::cout << "Extreme changes (>5%): " << extreme_changes << std::endl;
    std::cout << "===================" << std::endl << std::endl;
}

std::vector<double> generateGranularThresholds() {
    std::vector<double> thresholds;
    
    // Start from 0.05 and work down with increasingly fine granularity
    
    // First: 0.05 down to 0.01 (step by 0.01)
    for (double t = 0.05; t >= 0.01; t -= 0.01) {
        thresholds.push_back(t);
    }
    
    // Then: 0.009 down to 0.001 (step by 0.001)
    for (double t = 0.009; t >= 0.001; t -= 0.001) {
        thresholds.push_back(t);
    }
    
    // Then: 0.0009 down to 0.0001 (step by 0.0001)
    for (double t = 0.0009; t >= 0.0001; t -= 0.0001) {
        thresholds.push_back(t);
    }
    
    // Finally: 0.00009 down to 0.00001 (step by 0.00001)
    for (double t = 0.00009; t >= 0.00001; t -= 0.00001) {
        thresholds.push_back(t);
    }
    
    return thresholds;
}

int main() {
    std::string input_file = "../data/features.csv";
    std::string output_slide = "../output/sliding_anomalies.csv";
    std::string output_heap  = "../output/heap_anomalies.csv";

    std::vector<StockRow> data = read_features_csv(input_file);
    std::cout << "Loaded " << data.size() << " rows from " << input_file << "\n\n";

    if (data.empty()) {
        std::cerr << "No data loaded. Exiting." << std::endl;
        return 1;
    }

    std::vector<double> series;
    for (const auto& row : data) {
        series.push_back(row.daily_return);
    }

    analyzeData(series);
    
    int window_size = 30;
    
    // Sliding Window Detection
    double sliding_threshold = 2.5;
    std::cout << "=== SLIDING WINDOW DETECTION ===" << std::endl;
    std::cout << "Window size: " << window_size << std::endl;
    std::cout << "Threshold: " << sliding_threshold << " standard deviations" << std::endl;
    std::vector<int> sliding_flags = detectSlidingAnomalies(series, window_size, sliding_threshold);
    
    int sliding_count = 0;
    for (int flag : sliding_flags) sliding_count += flag;
    std::cout << "✅ Sliding window anomalies detected: " << sliding_count 
              << " (" << (100.0 * sliding_count / series.size()) << "%)" << std::endl << std::endl;

    // Heap-Based Detection - granular threshold search
    std::cout << "=== HEAP-BASED DETECTION (GRANULAR SEARCH) ===" << std::endl;
    
    auto heap_thresholds = generateGranularThresholds();
    std::cout << "Generated " << heap_thresholds.size() << " threshold values to test" << std::endl;
    std::cout << "Range: " << std::fixed << std::setprecision(6) << heap_thresholds[0] 
              << " down to " << heap_thresholds.back() << std::endl;
    std::cout << std::endl;
    
    std::vector<int> heap_flags;
    int heap_count = 0;
    double used_threshold = 0.0;
    const double TARGET_RATE = 0.05;  // Target 5% anomaly rate
    double best_rate_diff = 1.0;
    double best_threshold = heap_thresholds[0];
    int best_count = 0;
    
    for (size_t i = 0; i < heap_thresholds.size(); ++i) {
        double heap_threshold = heap_thresholds[i];
        
        std::cout << "[" << (i+1) << "/" << heap_thresholds.size() << "] "
                  << "Trying threshold " << std::fixed << std::setprecision(6) << heap_threshold << "..." << std::endl;
        
        std::vector<int> current_flags = detectHeapAnomalies(series, window_size, heap_threshold);
        
        int current_count = 0;
        for (int flag : current_flags) current_count += flag;
        
        double percentage = 100.0 * current_count / series.size();
        std::cout << "Found " << current_count << " anomalies (" 
                  << std::fixed << std::setprecision(4) << percentage << "%)" << std::endl;
        
        // Track the best threshold (closest to target rate)
        double rate_diff = std::abs(percentage/100.0 - TARGET_RATE);
        if (rate_diff < best_rate_diff) {
            best_rate_diff = rate_diff;
            best_threshold = heap_threshold;
            best_count = current_count;
            heap_flags = current_flags;  // Keep the best flags
        }
        
        // Check if this is good enough
        if (percentage >= 2.0 && percentage <= 8.0) {  // 2-8% range is reasonable
            used_threshold = heap_threshold;
            heap_count = current_count;
            heap_flags = current_flags;
            std::cout << "✅ Accepted! Good detection rate achieved." << std::endl;
            break;
        } else if (percentage > 20.0) {
            std::cout << "❌ Too many anomalies, trying lower threshold..." << std::endl;
        } else if (percentage < 0.5) {
            std::cout << "⚠️ Very few anomalies detected..." << std::endl;
        } else {
            std::cout << "🔍 Continuing search for optimal threshold..." << std::endl;
        }
        
        // Early stopping if we're getting very close to target
        if (percentage >= TARGET_RATE * 80 && percentage <= TARGET_RATE * 120) {
            used_threshold = heap_threshold;
            heap_count = current_count;
            heap_flags = current_flags;
            std::cout << "🎯 Close to target rate! Using this threshold." << std::endl;
            break;
        }
        
        std::cout << std::endl;
    }

    // If no threshold was explicitly accepted, use the best one found
    if (used_threshold == 0.0) {
        std::cout << "🎯 Using best threshold found: " << std::fixed << std::setprecision(6) << best_threshold << std::endl;
        used_threshold = best_threshold;
        heap_count = best_count;
        // heap_flags should already be set to the best flags
    }

    // Calculate overlap
    int overlap_count = 0;
    for (size_t i = 0; i < series.size(); ++i) {
        if (sliding_flags[i] == 1 && heap_flags[i] == 1) {
            overlap_count++;
        }
    }

    // Output results
    write_anomaly_output(output_slide, data, sliding_flags);
    write_anomaly_output(output_heap, data, heap_flags);

    std::cout << "\n===================================================" << std::endl;
    std::cout << "           FINAL RESULTS SUMMARY" << std::endl;
    std::cout << "===================================================" << std::endl;
    std::cout << "📊 Total data points analyzed: " << series.size() << std::endl;
    std::cout << "🔍 Sliding window anomalies: " << sliding_count 
              << " (" << std::fixed << std::setprecision(5) << (100.0 * sliding_count / series.size()) << "%)" << std::endl;
    std::cout << "🏔️  Heap-based anomalies: " << heap_count 
              << " (" << std::fixed << std::setprecision(5) << (100.0 * heap_count / series.size()) 
              << "%) [threshold: " << std::fixed << std::setprecision(6) << used_threshold << "]" << std::endl;
    std::cout << "🔄 Overlapping anomalies: " << overlap_count 
              << " (" << std::fixed << std::setprecision(5) << (100.0 * overlap_count / series.size()) << "%)" << std::endl;
    
    // Quality assessment
    double sliding_rate = 100.0 * sliding_count / series.size();
    double heap_rate = 100.0 * heap_count / series.size();
    
    std::cout << "\n📈 DETECTION QUALITY ASSESSMENT:" << std::endl;
    if (sliding_rate >= 1.0 && sliding_rate <= 5.0) {
        std::cout << "✅ Sliding window: Good detection rate" << std::endl;
    } else if (sliding_rate > 5.0) {
        std::cout << "⚠️ Sliding window: High detection rate (may be oversensitive)" << std::endl;
    } else {
        std::cout << "⚠️ Sliding window: Low detection rate (may be undersensitive)" << std::endl;
    }
    
    if (heap_rate >= 2.0 && heap_rate <= 8.0) {
        std::cout << "✅ Heap-based: Good detection rate" << std::endl;
    } else if (heap_rate > 8.0) {
        std::cout << "⚠️ Heap-based: High detection rate (may be oversensitive)" << std::endl;
    } else {
        std::cout << "⚠️ Heap-based: Low detection rate (may be undersensitive)" << std::endl;
    }
    
    std::cout << "\n📁 Results written to:" << std::endl;
    std::cout << "  • " << output_slide << std::endl;
    std::cout << "  • " << output_heap << std::endl;
    std::cout << "===================================================" << std::endl;

    return 0;
}