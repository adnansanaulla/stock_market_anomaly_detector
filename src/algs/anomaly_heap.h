#ifndef ANOMALY_HEAP_H
#define ANOMALY_HEAP_H

#include <vector>
#include <queue>
#include <iostream>
#include <iomanip>

/**
 * Detect anomalies using an improved heap-based approach with robust statistics
 * @param data: Input time series data
 * @param threshold: Threshold for anomaly detection (in terms of robust standard deviations)
 * @return: Vector of indices where anomalies were detected
 */
std::vector<int> detectAnomaliesHeap(const std::vector<double>& data, double threshold);

/**
 * Detect anomalies using granular threshold search to achieve target detection rate
 * @param data: Input time series data  
 * @param target_percentage: Target percentage of data points to flag as anomalies (default: 3%)
 * @return: Vector of indices where anomalies were detected
 */
std::vector<int> detectAnomaliesHeapGranular(const std::vector<double>& data, 
                                           double target_percentage = 0.03);

#endif // ANOMALY_HEAP_H