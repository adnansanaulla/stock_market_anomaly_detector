#include "csv_utils.h"
#include <fstream>
#include <sstream>
#include <iostream>

std::vector<StockRow> read_features_csv(const std::string& filename) {
    std::vector<StockRow> data;
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << "\n";
        return data;
    }

    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        StockRow row;

        std::getline(ss, row.date, ',');
        std::getline(ss, cell, ','); row.open = std::stod(cell);
        std::getline(ss, cell, ','); row.high = std::stod(cell);
        std::getline(ss, cell, ','); row.low = std::stod(cell);
        std::getline(ss, cell, ','); row.close = std::stod(cell);
        std::getline(ss, cell, ','); row.adj_close = std::stod(cell);
        std::getline(ss, cell, ','); row.volume = std::stod(cell);
        std::getline(ss, row.ticker, ',');
        std::getline(ss, cell, ','); row.daily_return = std::stod(cell);
        std::getline(ss, cell, ','); row.volatility = std::stod(cell);
        std::getline(ss, cell, ','); row.volume_zscore = std::stod(cell);

        data.push_back(row);
    }

    return data;
}

void write_anomaly_output(const std::string& filename, const std::vector<StockRow>& data, const std::vector<int>& flags) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to write to: " << filename << "\n";
        return;
    }

    file << "Date,Ticker,Open,High,Low,Close,Adj Close,Volume,Daily Return,Volatility,Volume Z-Score,Anomaly\n";

    for (size_t i = 0; i < data.size(); ++i) {
        const auto& row = data[i];
        file << row.date << "," << row.ticker << ","
             << row.open << "," << row.high << "," << row.low << ","
             << row.close << "," << row.adj_close << "," << row.volume << ","
             << row.daily_return << "," << row.volatility << "," << row.volume_zscore << ","
             << flags[i] << "\n";
    }

    file.close();
}