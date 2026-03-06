#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <omp.h>
#include <windows.h>

using namespace std;

vector<vector<double>> readMatrix(const string& filename, int& n) {
    ifstream file(filename);

    if (!file) {
        cerr << "Error opening file " << filename << endl;
        exit(1);
    }

    file >> n;
    vector<vector<double>> m(n, vector<double>(n));

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            file >> m[i][j];

    return m;
}

void setCoreAffinity(int cores) {
    DWORD_PTR mask = 0;

    for (int i = 0; i < cores; i++) {
        mask |= (1ULL << i);
    }

    SetProcessAffinityMask(GetCurrentProcess(), mask);
}

int main() {

    SetConsoleCP(65001);
    SetConsoleOutputCP(65001);

    int threads;
    int cores;

    cout << "Available logical processors: " << omp_get_num_procs() << endl;

    cout << "Enter number of threads: ";
    cin >> threads;

    cout << "Enter number of cores to use: ";
    cin >> cores;

    omp_set_num_threads(threads);
    setCoreAffinity(cores);

    int n1, n2;

    auto A = readMatrix("matrixA.txt", n1);
    auto B = readMatrix("matrixB.txt", n2);

    if (n1 != n2) {
        cerr << "Matrix sizes do not match!" << endl;
        return 1;
    }

    int n = n1;

    vector<vector<double>> C(n, vector<double>(n, 0));

    auto start = chrono::high_resolution_clock::now();

#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {

            double sum = 0;

            for (int k = 0; k < n; k++)
                sum += A[i][k] * B[k][j];

            C[i][j] = sum;
        }

    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> elapsed = end - start;

    cout << endl;
    cout << "Matrix size: " << n << "x" << n << endl;
    cout << "Threads: " << threads << endl;
    cout << "Cores used: " << cores << endl;
    cout << "Execution time: " << elapsed.count() << " seconds" << endl;

    return 0;
}