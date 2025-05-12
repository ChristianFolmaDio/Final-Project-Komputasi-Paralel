#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>

void seq_gen(int n, char seq[]) {
    for (int i = 0; i < n; i++) {
        int base = rand() % 4;
        switch (base) {
            case 0: seq[i] = 'a'; break;
            case 1: seq[i] = 't'; break;
            case 2: seq[i] = 'c'; break;
            case 3: seq[i] = 'g'; break;
        }
    }
}

int main() {
    srand(static_cast<unsigned int>(time(0))); // Seed random number generator

    int sequence_length = 100000; // Ubah sesuai kebutuhan
    char* sequence = new char[sequence_length + 1];
    sequence[sequence_length] = '\0'; // Null-terminate string

    seq_gen(sequence_length, sequence);

    // Simpan ke file
    std::ofstream outfile("random_dna_sequence.txt");
    if (!outfile) {
        std::cerr << "Error membuka file untuk menulis." << std::endl;
        delete[] sequence;
        return 1;
    }

    outfile << sequence;
    outfile.close();

    std::cout << "Sekuens DNA acak dengan panjang " << sequence_length << " telah disimpan ke random_dna_sequence.txt" << std::endl;

    delete[] sequence;
    return 0;
}
