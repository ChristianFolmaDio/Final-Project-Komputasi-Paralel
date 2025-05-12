#include <iostream>
#include <cstring>
#include <fstream>
#include <vector>
#include <algorithm>
#include "time.h"
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

using namespace std;

// Fungsi untuk memangkas whitespace dan newline di awal dan akhir string
string trim(const string& str) {
    size_t first = str.find_first_not_of(" \n\r\t");
    if (first == string::npos) return "";
    size_t last = str.find_last_not_of(" \n\r\t");
    return str.substr(first, (last - first + 1));
}

// Fungsi untuk menghapus semua newline dan carriage return dari string
void removeNewlines(string& str) {
    str.erase(remove(str.begin(), str.end(), '\n'), str.end());
    str.erase(remove(str.begin(), str.end(), '\r'), str.end());
}

void preKMP(char* pattern, int f[], int n) {
    int k;
    f[0] = -1;
    for (int i = 1; i < n; i++) {
        k = f[i - 1];
        while (k >= 0) {
            if (pattern[k] == pattern[i - 1])
                break;
            else
                k = f[k];
        }
        f[i] = k + 1;
    }
}

__global__ void KMP(char* pattern, char* target, int f[], int c[], int n, int m) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int i = index;  // each thread starts at a different position
    if (i > m - n) return; // prevent out of bounds

    int k = 0;
    int pos = i;
    while (pos < m) {
        if (k == -1) {
            pos++;
            k = 0;
        }
        else if (target[pos] == pattern[k]) {
            pos++;
            k++;
            if (k == n) {
                c[i] = n; // mark match at start index i
                break;  // found a match, exit
            }
        }
        else {
            k = f[k];
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <data_file> <pattern_file>" << endl;
        return 1;
    }

    // Start CUDA profiler
    cudaProfilerStart();

    // Baca file
    ifstream f_data(argv[1]);
    ifstream f_pattern(argv[2]);
    ofstream f2("output.txt");

    if (!f_data.is_open() || !f_pattern.is_open()) {
        cerr << "Error opening input files." << endl;
        return 1;
    }

    // Baca seluruh isi file dan trim whitespace/newline
    string data_str((istreambuf_iterator<char>(f_data)), istreambuf_iterator<char>());
    string pattern_str((istreambuf_iterator<char>(f_pattern)), istreambuf_iterator<char>());

    data_str = trim(data_str);
    pattern_str = trim(pattern_str);

    // Hapus newline dan carriage return dari data dan pola
    removeNewlines(data_str);
    removeNewlines(pattern_str);

    // Ubah ke uppercase agar pencocokan tidak case sensitive
    transform(data_str.begin(), data_str.end(), data_str.begin(), ::toupper);
    transform(pattern_str.begin(), pattern_str.end(), pattern_str.begin(), ::toupper);

    int m = data_str.size();
    int n = pattern_str.size();

    cout << "Panjang data: " << m << ", panjang pola: " << n << endl;

    // Alokasi memori di host
    vector<char> tar(data_str.begin(), data_str.end());
    vector<char> pat(pattern_str.begin(), pattern_str.end());
    vector<int> c(m, -1);
    vector<int> f(n);

    preKMP(pat.data(), f.data(), n);

    // Alokasi memori di device
    char *d_tar, *d_pat;
    int *d_c, *d_f;

    cout << "----Start copying data to GPU----" << endl;
    cudaEvent_t copy_start, copy_stop;
    CUDA_CHECK(cudaEventCreate(&copy_start));
    CUDA_CHECK(cudaEventCreate(&copy_stop));
    CUDA_CHECK(cudaEventRecord(copy_start));

    CUDA_CHECK(cudaMalloc((void**)&d_tar, m));
    CUDA_CHECK(cudaMalloc((void**)&d_pat, n));
    CUDA_CHECK(cudaMalloc((void**)&d_f, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_c, m * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_tar, tar.data(), m, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pat, pat.data(), n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_f, f.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, c.data(), m * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(copy_stop));
    CUDA_CHECK(cudaEventSynchronize(copy_stop));
    float copy_time;
    CUDA_CHECK(cudaEventElapsedTime(&copy_time, copy_start, copy_stop));
    cout << "----Data copied to GPU successfully---- Takes " << copy_time / 1000 << " seconds" << endl;

    // Jalankan KMP di GPU
    int M = 1024;
    if (n > 10000000) M = 128;

    dim3 blocks((m + M - 1) / M);
    dim3 threads(M);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    KMP<<<blocks, threads>>>(d_pat, d_tar, d_f, d_c, n, m);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float time_elapsed;
    CUDA_CHECK(cudaEventElapsedTime(&time_elapsed, start, stop));
    cout << "----String matching done---- Takes " << time_elapsed / 1000 << " seconds" << endl;

    // Copy hasil kembali ke host
    CUDA_CHECK(cudaMemcpy(c.data(), d_c, m * sizeof(int), cudaMemcpyDeviceToHost));

    // Tulis output dan tampilkan hasil
    bool found = false;
    for (int i = 0; i < m; i++) {
        if (c[i] != -1) {
            f2 << i << ' ' << c[i] << '\n';
            cout << "Pola ditemukan di posisi " << i << " dengan panjang " << c[i] << endl;
            found = true;
        }
    }
    if (!found) {
        // cout << "Tidak ada pola yang ditemukan dalam data." << endl;
    }

    // Bersihkan memori
    CUDA_CHECK(cudaFree(d_tar));
    CUDA_CHECK(cudaFree(d_pat));
    CUDA_CHECK(cudaFree(d_f));
    CUDA_CHECK(cudaFree(d_c));

    // Stop CUDA profiler
    cudaProfilerStop();

    return 0;
}
