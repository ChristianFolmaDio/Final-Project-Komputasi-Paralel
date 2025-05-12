# Analisis Perbandingan Proses Paralel antara Algoritma Aho-Corasick dan Knuth-Morris Pratt Menggunakan CUDA untuk Pencocokan Urutan DNA
1. Deskripsi Proyek
   
Proyek ini bertujuan menganalisis dan membandingkan performa implementasi algoritma Knuth-Morris Pratt (KMP) dan Aho-Corasick (AC) dalam pencocokan string, khususnya untuk pencarian pola pada urutan DNA. Fokus utama diberikan pada implementasi paralel menggunakan CUDA (GPU) dibandingkan dengan versi sekuensial di CPU.

Eksperimen dilakukan menggunakan dataset DNA sintetik sepanjang 100.000 karakter, dengan variasi jumlah pola (pattern) dari 8 hingga 1024. Meskipun algoritma KMP berhasil diimplementasikan dalam versi CPU dan GPU, implementasi CUDA untuk Aho-Corasick gagal diselesaikan karena kendala teknis dan kompatibilitas dengan compiler CUDA.

2. Algoritma

- Knuth-Morris Pratt (KMP):

Versi CPU: Implementasi sekuensial dengan pre-processing (prefix function), pencocokan, dan pengukuran waktu dengan chrono.

Versi CUDA: Pencocokan dilakukan oleh thread GPU secara paralel. Menggunakan std::vector untuk manajemen memori dan cudaEvent untuk pengukuran waktu.

- Aho-Corasick:

Direncanakan untuk diuji, namun implementasi CUDA tidak dapat diselesaikan karena kompleksitas struktur trie dan isu kompatibilitas compiler.

3. Dataset

- DNA sequence sepanjang 100.000 karakter, dihasilkan secara acak menggunakan program C++.

- Pola pencarian (pattern) dengan panjang tetap 10 karakter.

- Jumlah pola: 8, 16, 32, 64, 128, 256, 512, 1024.

4. Kesimpulan dan Hasil

Berdasarkan seluruh data ujicoba dan hasil profiling, dapat disimpulkan bahwa:

- Implementasi KMP Sekuensial di CPU Jauh Lebih Unggul: Untuk tugas pencocokan string KMP pada data berukuran 100.000 karakter dan pola dalam rentang yang diuji, implementasi sekuensial di CPU memberikan performa yang jauh lebih cepat dibandingkan dengan implementasi paralel di GPU menggunakan CUDA.

- Overhead GPU Melebihi Manfaat Paralelisasi: Kesenjangan performa yang besar menunjukkan bahwa overhead terkait penggunaan GPU (seperti waktu peluncuran dan manajemen kernel, sinkronisasi antar tahapan, dan potensi idle time atau inefisiensi dalam eksekusi kernel paralel pada skala ini) jauh lebih besar daripada penghematan waktu komputasi yang didapat dari eksekusi paralel, jika dibandingkan dengan kecepatan tinggi dari algoritma KMP sekuensial yang efisien di CPU modern.

- Efisiensi Implementasi GPU Bergantung pada Panjang Pola: Performasi implementasi KMP CUDA Anda bervariasi dengan panjang pola. Pola 128 memberikan performa terbaik di GPU dalam rentang yang diuji, menunjukkan bahwa pada panjang pola tersebut, keseimbangan antara kerja komputasi paralel dan overhead relatif paling baik. Peningkatan atau penurunan performa di luar titik ini mengindikasikan bahwa cara implementasi CUDA menangani pola yang lebih pendek atau lebih panjang kurang optimal dibandingkan pada pola 128.

- Analisis Profil Nsight Penting: Profil Nsight Systems sangat membantu dalam memahami mengapa GPU lebih lambat. Ia mengungkapkan bahwa eksekusi GPU tidak mulus dalam satu blok komputasi, melainkan terbagi dalam beberapa tahap, yang menambah overhead. Analisis lebih mendalam pada metrik spesifik di Nsight (misalnya, thread utilization, memory access patterns, kernel launch latencies) diperlukan untuk mengidentifikasi akar penyebab inefisiensi di dalam kernel itu sendiri.

Secara umum, hasil ini menunjukkan bahwa memindahkan algoritma ke GPU tidak secara otomatis menghasilkan percepatan. Ini sangat bergantung pada sifat algoritma (seberapa baik ia dapat diparalelkan secara masif tanpa banyak ketergantungan data), kualitas implementasi paralel, dan skala masalah. Untuk tugas yang sudah sangat cepat di CPU sekuensial (seperti KMP pada data berukuran sedang), overhead penggunaan GPU bisa menjadi penghalang utama performa.
Eksperimen ini memberikan data yang berharga tentang performa implementasi spesifik Anda dan menyoroti pentingnya profiling untuk memahami bottleneck dalam komputasi paralel. Untuk melihat potensi keuntungan GPU untuk KMP, pengujian pada skala data yang jauh lebih besar mungkin diperlukan, bersamaan dengan optimasi mendalam pada kode kernel CUDA berdasarkan wawasan dari Nsight Systems.

Aho-Corasick

Rencana awal penelitian kami meliputi komparasi performa antara algoritma Knuth-Morris-Pratt (KMP) dan algoritma Aho-Corasick. Namun, kami menghadapi kendala teknis yang signifikan dalam proses implementasi algoritma Aho-Corasick, termasuk kesulitan pada tahap coding serta keterbatasan atau isu kompatibilitas dengan compiler yang kami gunakan. Oleh karena itu, kami tidak dapat menyelesaikan ujicoba performa untuk algoritma Aho-Corasick, dan analisis perbandingan dalam laporan ini hanya berfokus pada algoritma KMP.
