#include <mpi.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <unistd.h>
#include <sched.h>
#include <cstring>

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

void setCoreAffinityByRank(int rank) {
    int numCores = sysconf(_SC_NPROCESSORS_ONLN);
    if (numCores <= 0) return;

    int coreToUse = rank % numCores;
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(coreToUse, &mask);

    sched_setaffinity(0, sizeof(mask), &mask);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    int activeProcesses = worldSize;

    int matrixSize = 1000;
    if (argc > 1) {
        matrixSize = atoi(argv[1]);
    }

    if (rank == 0) {
        int numCores = sysconf(_SC_NPROCESSORS_ONLN);
        cout << "========================================" << endl;
        cout << "MPI Matrix Multiplication" << endl;
        cout << "========================================" << endl;
        cout << "Available cores: " << numCores << endl;
        cout << "MPI processes used: " << worldSize << endl;
        cout << "Matrix size: " << matrixSize << "x" << matrixSize << endl;
        cout << "========================================" << endl;
    }

    setCoreAffinityByRank(rank);

    int n = matrixSize;

    if (n % activeProcesses != 0) {
        if (rank == 0) {
            cerr << "Error: Matrix size (" << n << ") must be divisible by number of processes (" << activeProcesses << ")" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    vector<double> flatA, flatB, flatC;

    if (rank == 0) {
        int n1, n2;
        vector<vector<double>> A = readMatrix("matrixA.txt", n1);
        vector<vector<double>> B = readMatrix("matrixB.txt", n2);

        if (n1 != n2 || n1 != n) {
            cerr << "Matrix size mismatch!" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        flatA = flattenMatrix(A);
        flatB = flattenMatrix(B);
        flatC.resize(n * n);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

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

    for (int i = 0; i < rowsPerProcess; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += localA[i * n + k] * flatB[k * n + j];
            }
            localC[i * n + j] = sum;
        }
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
        cout << "========================================" << endl;
        cout << "Execution time: " << elapsed.count() << " seconds" << endl;
        cout << "========================================" << endl;
    }

    MPI_Finalize();
    return 0;
}