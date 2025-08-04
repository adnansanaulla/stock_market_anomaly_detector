#include <iostream>
#include <vector>
#include <string>

#include "utils/csv_utils.h"
#include "algs/anomaly_sliding_window.h"
#include "algs/anomaly_heap.h"

int main() {
    std::string input_file = "../data/features.csv";
    std::string output_slide = "../output/sliding_anomalies.csv";
    std::string output_heap  = "../output/heap_anomalies.csv";

    // read data
    std::vector<StockRow> data = read_features_csv(input_file);
    std::cout << "Loaded " << data.size() << " rows from " << input_file << "\n";

    // Extract the daily return column
    std::vector<double> series;
    for (const auto& row : data) {
        series.push_back(row.daily_return);
    }

    // Apply Sliding Window Detection
    int window_size = 30;
    double sliding_threshold = 2.0;
    std::vector<int> sliding_flags = detectSlidingAnomalies(series, window_size, sliding_threshold);

    // Apply Heap-Based Detection
    double heap_threshold = 5.0;
    std::vector<int> heap_flags = detectHeapAnomalies(series, window_size, heap_threshold);

    // Output results
    write_anomaly_output(output_slide, data, sliding_flags);
    write_anomaly_output(output_heap, data, heap_flags);

    std::cout << "Anomaly results written to:\n";
    std::cout << "  -> " << output_slide << "\n";
    std::cout << "  -> " << output_heap << "\n";

    return 0;
}
