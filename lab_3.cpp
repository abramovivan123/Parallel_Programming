#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <mpi.h>
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

vector<double> flattenMatrix(const vector<vector<double>>& matrix) {
    int n = matrix.size();
    vector<double> flat(n * n);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            flat[i * n + j] = matrix[i][j];

    return flat;
}

vector<vector<double>> unflattenMatrix(const vector<double>& flat, int n) {
    vector<vector<double>> matrix(n, vector<double>(n));

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            matrix[i][j] = flat[i * n + j];

    return matrix;
}

void setCoreAffinityByRank(int rank) {
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);

    int logicalProcessors = sysInfo.dwNumberOfProcessors;
    if (logicalProcessors <= 0) return;

    int coreToUse = rank % logicalProcessors;
    DWORD_PTR mask = (1ULL << coreToUse);

    SetProcessAffinityMask(GetCurrentProcess(), mask);
}

int main(int argc, char* argv[]) {

    SetConsoleCP(65001);
    SetConsoleOutputCP(65001);

    MPI_Init(&argc, &argv);

    int rank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    int activeProcesses = 1;

    if (rank == 0) {
        SYSTEM_INFO sysInfo;
        GetSystemInfo(&sysInfo);

        cout << "Available logical processors: " << sysInfo.dwNumberOfProcessors << endl;
        cout << "Started MPI processes: " << worldSize << endl;
        cout << "Enter number of processes to use: ";
        cin >> activeProcesses;

        if (activeProcesses < 1)
            activeProcesses = 1;

        if (activeProcesses > worldSize) {
            cout << "Requested number is greater than started MPI processes." << endl;
            cout << "Using " << worldSize << " processes." << endl;
            activeProcesses = worldSize;
        }
    }

    MPI_Bcast(&activeProcesses, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank >= activeProcesses) {
        MPI_Finalize();
        return 0;
    }

    setCoreAffinityByRank(rank);

    int n1 = 0, n2 = 0, n = 0;
    vector<vector<double>> A, B;
    vector<double> flatA, flatB, flatC;

    if (rank == 0) {
        A = readMatrix("matrixA.txt", n1);
        B = readMatrix("matrixB.txt", n2);

        if (n1 != n2) {
            cerr << "Matrix sizes do not match!" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        n = n1;
        flatA = flattenMatrix(A);
        flatB = flattenMatrix(B);
        flatC.resize(n * n);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (n % activeProcesses != 0) {
        if (rank == 0) {
            cerr << "For this version, matrix size must be divisible by number of active processes!" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    int rowsPerProcess = n / activeProcesses;

    vector<double> localA(rowsPerProcess * n);
    vector<double> localC(rowsPerProcess * n, 0.0);

    if (rank != 0) {
        flatB.resize(n * n);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = chrono::high_resolution_clock::now();

    MPI_Scatter(
        rank == 0 ? flatA.data() : nullptr,
        rowsPerProcess * n,
        MPI_DOUBLE,
        localA.data(),
        rowsPerProcess * n,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    MPI_Bcast(flatB.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < rowsPerProcess; i++)
        for (int j = 0; j < n; j++) {

            double sum = 0.0;

            for (int k = 0; k < n; k++)
                sum += localA[i * n + k] * flatB[k * n + j];

            localC[i * n + j] = sum;
        }

    MPI_Gather(
        localC.data(),
        rowsPerProcess * n,
        MPI_DOUBLE,
        rank == 0 ? flatC.data() : nullptr,
        rowsPerProcess * n,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = chrono::high_resolution_clock::now();

    if (rank == 0) {
        chrono::duration<double> elapsed = end - start;

        cout << endl;
        cout << "Matrix size: " << n << "x" << n << endl;
        cout << "MPI processes used: " << activeProcesses << endl;
        cout << "Execution time: " << elapsed.count() << " seconds" << endl;
    }

    MPI_Finalize();
    return 0;
}