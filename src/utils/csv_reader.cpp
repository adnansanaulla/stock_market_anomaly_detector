// #include <fstream>
// #include <sstream>
// using namespace std;
//
// vector<vector<double>> read_csv(const string& filename) {
//     vector<vector<double>> data;
//     ifstream file(filename);
//     string line;
//
//     getline(file, line); // skip header
//     while (getline(file, line)) {
//         stringstream ss(line);
//         string cell;
//         vector<double> row;
//         while (getline(ss, cell, ',')) row.push_back(stod(cell));
//         data.push_back(row);
//     }
//     return data;
// }
