#include "utils/csv_utils.h"
#include <iostream>

int main() {
    std::string filename = "../../data/features.csv";

    std::vector<StockRow> rows = read_features_csv(filename);

    std::cout << "Loaded " << rows.size() << " rows from " << filename << "\n";

    // Print the first 3 rows
    for (int i = 0; i < std::min(3, (int)rows.size()); ++i) {
        const auto& row = rows[i];
        std::cout << row.date << " | "
                  << row.ticker << " | "
                  << row.close << " | "
                  << row.daily_return << " | "
                  << row.volume_zscore << "\n";
    }

    return 0;
}