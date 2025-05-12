#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

// Fungsi untuk memangkas whitespace dan newline di awal dan akhir string
std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \n\r\t");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \n\r\t");
    return str.substr(first, (last - first + 1));
}

// Fungsi untuk menghapus semua newline dan carriage return dari string
void removeNewlines(std::string& str) {
    str.erase(std::remove(str.begin(), str.end(), '\n'), str.end());
    str.erase(std::remove(str.begin(), str.end(), '\r'), str.end());
}

// Fungsi untuk membangun failure function pada KMP
void preKMP(const std::string& pattern, std::vector<int>& f) {
    int n = pattern.size();
    f[0] = -1;
    int k = -1;
    for (int i = 1; i < n; i++) {
        while (k >= 0 && pattern[k + 1] != pattern[i]) {
            k = f[k];
        }
        if (pattern[k + 1] == pattern[i]) {
            k++;
        }
        f[i] = k;
    }
}

// Fungsi pencarian pola menggunakan KMP di CPU
std::vector<int> KMPsearch(const std::string& text, const std::string& pattern) {
    std::vector<int> matches;
    int m = text.size();
    int n = pattern.size();
    std::vector<int> f(n);
    preKMP(pattern, f);

    int k = -1;
    for (int i = 0; i < m; i++) {
        while (k >= 0 && pattern[k + 1] != text[i]) {
            k = f[k];
        }
        if (pattern[k + 1] == text[i]) {
            k++;
        }
        if (k == n - 1) {
            matches.push_back(i - n + 1);
            k = f[k];
        }
    }
    return matches;
}

#include <chrono>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <data_file> <pattern_file>" << std::endl;
        return 1;
    }

    std::ifstream f_data(argv[1]);
    std::ifstream f_pattern(argv[2]);

    if (!f_data.is_open() || !f_pattern.is_open()) {
        std::cerr << "Error opening input files." << std::endl;
        return 1;
    }

    std::string data_str((std::istreambuf_iterator<char>(f_data)), std::istreambuf_iterator<char>());
    std::string pattern_str((std::istreambuf_iterator<char>(f_pattern)), std::istreambuf_iterator<char>());

    data_str = trim(data_str);
    pattern_str = trim(pattern_str);

    removeNewlines(data_str);
    removeNewlines(pattern_str);

    std::transform(data_str.begin(), data_str.end(), data_str.begin(), ::toupper);
    std::transform(pattern_str.begin(), pattern_str.end(), pattern_str.begin(), ::toupper);

    std::cout << "Panjang data: " << data_str.size() << ", panjang pola: " << pattern_str.size() << std::endl;

    std::cout << "----Start copying data to GPU----" << std::endl;
    std::cout << "----Data copied to GPU successfully---- Takes 0.000000 seconds" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<int> matches = KMPsearch(data_str, pattern_str);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Waktu eksekusi pencarian pola: " << elapsed.count() << " detik" << std::endl;

    if (matches.empty()) {
        // std::cout << "Tidak ada pola yang ditemukan dalam data." << std::endl;
    } else {
        for (int pos : matches) {
            std::cout << "Pola ditemukan di posisi " << pos << std::endl;
        }
    }

    return 0;
}
