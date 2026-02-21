#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <locale>

using namespace std;

vector<vector<double>> readMatrix(const string& filename, int &n) {
    ifstream file(filename);
    if (!file) {
        cerr << "Ошибка открытия файла " << filename << endl;
        exit(1);
    }

    file >> n;
    vector<vector<double>> m(n, vector<double>(n));

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            file >> m[i][j];

    return m;
}

void writeMatrix(const string& filename, const vector<vector<double>>& m) {
    ofstream file(filename);
    int n = m.size();
    file << n << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            file << m[i][j] << " ";
        file << endl;
    }
}

int main() {
    setlocale(LC_ALL, ".UTF-8");

    int n1, n2;

    auto A = readMatrix("matrixA.txt", n1);
    auto B = readMatrix("matrixB.txt", n2);

    if (n1 != n2) {
        cerr << "Размеры матриц не совпадают!" << endl;
        return 1;
    }

    int n = n1;
    vector<vector<double>> C(n, vector<double>(n, 0));

    auto start = chrono::high_resolution_clock::now();

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                C[i][j] += A[i][k] * B[k][j];

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    writeMatrix("result.txt", C);

    cout << "Размер матриц: " << n << "x" << n << endl;
    cout << "Объем задачи (операций умножения): " << n*n*n << endl;
    cout << "Время выполнения: " << elapsed.count() << " секунд" << endl;

    return 0;
}