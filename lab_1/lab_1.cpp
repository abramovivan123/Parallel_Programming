#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <string>
#include <filesystem>
#include <iomanip>
#include <locale>

using namespace std;
namespace fs = std::filesystem;

static vector<vector<double>> readMatrix(const fs::path& filename, int& n) {
    ifstream file(filename);
    if (!file) {
        cerr << "Ошибка открытия файла: " << filename.string() << "\n";
        throw runtime_error("Не удалось открыть входной файл");
    }

    file >> n;
    if (!file || n <= 0) {
        cerr << "Некорректный формат файла (не удалось прочитать N): " << filename.string() << "\n";
        throw runtime_error("Некорректный формат входного файла");
    }

    vector<vector<double>> m(n, vector<double>(n));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            file >> m[i][j];
            if (!file) {
                cerr << "Некорректные данные матрицы в файле: " << filename.string()
                    << " (строка " << (i + 2) << ", столбец " << (j + 1) << ")\n";
                throw runtime_error("Некорректные данные матрицы");
            }
        }
    }

    return m;
}

static void writeMatrix(const fs::path& filename, const vector<vector<double>>& m) {
    ofstream file(filename);
    if (!file) {
        cerr << "Ошибка записи файла: " << filename.string() << "\n";
        throw runtime_error("Не удалось открыть файл для записи");
    }

    int n = (int)m.size();
    file << n << "\n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            file << m[i][j];
            if (j + 1 < n) file << ' ';
        }
        file << "\n";
    }
}

static vector<vector<double>> multiply(const vector<vector<double>>& A,
    const vector<vector<double>>& B) {
    int n = (int)A.size();
    vector<vector<double>> C(n, vector<double>(n, 0.0));

    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) {
            const double aik = A[i][k];
            for (int j = 0; j < n; j++)
                C[i][j] += aik * B[k][j];
        }

    return C;
}

int main() {
    setlocale(LC_ALL, ".UTF-8");

    const std::filesystem::path BASE_DIR =
    std::filesystem::path(PROJECT_DIR) / u8"перемножение матриц разных размеров";

    const int sizes[] = { 200, 400, 800, 1200, 1600 };

    cout << "Текущая рабочая папка: " << fs::current_path().string() << "\n";
    cout << "Базовая папка с матрицами: " << BASE_DIR.string() << "\n\n";

    struct Row { int n; double sec; long long ops; };
    vector<Row> results;

    for (int nExpected : sizes) {
        try {
            fs::path dir = BASE_DIR / (to_string(nExpected) + "x" + to_string(nExpected));
            fs::path fileA = dir / "matrixA.txt";
            fs::path fileB = dir / "matrixB.txt";
            fs::path fileC = dir / "result.txt";

            int n1 = 0, n2 = 0;
            auto A = readMatrix(fileA, n1);
            auto B = readMatrix(fileB, n2);

            if (n1 != n2) {
                cerr << "Ошибка: размеры матриц не совпадают в папке " << dir.string()
                    << " (A: " << n1 << ", B: " << n2 << ")\n";
                return 1;
            }
            if (n1 != nExpected) {
                cerr << "Предупреждение: в " << dir.string()
                    << " ожидается N=" << nExpected << ", но в файле N=" << n1 << "\n";
            }

            auto start = chrono::high_resolution_clock::now();
            auto C = multiply(A, B);
            auto end = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = end - start;

            writeMatrix(fileC, C);

            long long ops = 1LL * n1 * n1 * n1;
            results.push_back({ n1, elapsed.count(), ops });

            cout << "OK: " << n1 << "x" << n1
                << " | ops=" << ops
                << " | time=" << fixed << setprecision(6) << elapsed.count() << " сек"
                << " | saved: " << fileC.string() << "\n";
        }
        catch (const exception& e) {
            cerr << "\nСбой при обработке N=" << nExpected << ": " << e.what() << "\n";
            cerr << "Проверь, что файлы существуют по пути:\n"
                << "  " << (BASE_DIR / (to_string(nExpected) + "x" + to_string(nExpected)) / "matrixA.txt").string() << "\n"
                << "  " << (BASE_DIR / (to_string(nExpected) + "x" + to_string(nExpected)) / "matrixB.txt").string() << "\n";
            return 1;
        }
    }

    cout << "\n=== Итоговая таблица ===\n";
    cout << left << setw(10) << "N"
        << setw(18) << "ops (N^3)"
        << "time (sec)\n";
    for (const auto& r : results) {
        cout << left << setw(10) << r.n
            << setw(18) << r.ops
            << fixed << setprecision(6) << r.sec << "\n";
    }

    return 0;
}