# stock_market_anomaly_detector

# Step 1 Part 1: Have a c++ compiler that supports c++11 & have Python 3.7 or newer 
# Step 1 Part 2: Install these required packages: pip install pandas numpy matplotlib seaborn yfinance pathlib

# Step 2: Clone the repository to whatever file you like
# Example: git clone https://github.com/adnansanaulla/stock_market_anomaly_detector.git
# Step 2 Part 2: Delete all the files inside plots folder if you want to see the files be generated 

# Step 3 Part 1: change the directory to the src folder
# Step 3 Part 2: in the terminal call: python fetch_stock_data.py
# Explanation: This downloads all the stock data

# Step 3 Part 3: in the terminal call: python preprocess_features.py
# Explanation: This generates the specific features associated with the data set

# Step 4: Compile the c++ code in the src directory of the terminal
# Use this: g++ -std=c++17 -O2 -o main main.cpp utils/csv_utils.cpp utils/rolling_stats.cpp algs/anomaly_sliding_window.cpp algs/anomaly_heap.cpp

# So once you do that you can then call: .\main
# Result: This runs the stock market anomaly detection pipeline that's coded in main.cpp

# Step 5: in the src directory call this: python anomaly_comparison.py
# Explanation: This generates plots comparing detected anomalies using matplotlib & seaborn