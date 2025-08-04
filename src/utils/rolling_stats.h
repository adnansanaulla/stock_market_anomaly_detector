#pragma once
#include <deque>

class RollingStats {
public:
    RollingStats(int window_size);

    void add(double value);
    double mean() const;
    double stddev() const;
    bool ready() const;

private:
    int window_size;
    std::deque<double> window;
};
