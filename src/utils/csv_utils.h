#ifndef CSV_UTILS_H
#define CSV_UTILS_H

#include <string>
#include <vector>

// represents a row from features.csv
struct StockRow {
    std::string date;
    double open;
    double high;
    double low;
    double close;
    double adj_close;
    double volume;
    std::string ticker;
    double daily_return;
    double volatility;
    double volume_zscore;
};

// reads the features.csv file
std::vector<StockRow> read_features_csv(const std::string& filename);

// writes a new CSV that includes an anomaly flag column
void write_anomaly_output(const std::string& filename, const std::vector<StockRow>& data, const std::vector<int>& flags);

// ADD THIS LINE - loads CSV data into a simple vector of doubles for anomaly detection
bool loadCSV(const std::string& filename, std::vector<double>& data);

#endif // CSV_UTILS_H