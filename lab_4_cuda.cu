#include <cuda_runtime.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace std;

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err__ = (call);                                                   \
        if (err__ != cudaSuccess) {                                                   \
            throw runtime_error(string("CUDA error: ") + cudaGetErrorString(err__) + \
                                " at " + __FILE__ + ":" + to_string(__LINE__));      \
        }                                                                             \
    } while (0)

vector<vector<double>> readMatrix(const string& filename, int& n) {
    ifstream file(filename);
    if (!file) {
        throw runtime_error("Error opening file " + filename);
    }

    file >> n;
    if (n <= 0) {
        throw runtime_error("Invalid matrix size in file " + filename);
    }

    vector<vector<double>> matrix(n, vector<double>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (!(file >> matrix[i][j])) {
                throw runtime_error("Not enough matrix elements in file " + filename);
            }
        }
    }

    return matrix;
}

vector<double> flattenMatrix(const vector<vector<double>>& matrix) {
    const int n = static_cast<int>(matrix.size());
    vector<double> flat(static_cast<size_t>(n) * n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            flat[static_cast<size_t>(i) * n + j] = matrix[i][j];
        }
    }

    return flat;
}

__global__ void multiplyMatricesKernel(const double* A,
                                       const double* B,
                                       double* C,
                                       int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        double sum = 0.0;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[static_cast<size_t>(k) * n + col];
        }
        C[static_cast<size_t>(row) * n + col] = sum;
    }
}

int main() {
    try {
        int n1 = 0;
        int n2 = 0;

        const auto A = readMatrix("matrixA.txt", n1);
        const auto B = readMatrix("matrixB.txt", n2);

        if (n1 != n2) {
            cerr << "Matrix sizes do not match!" << endl;
            return 1;
        }

        const int n = n1;
        const size_t bytes = static_cast<size_t>(n) * n * sizeof(double);

        const vector<double> flatA = flattenMatrix(A);
        const vector<double> flatB = flattenMatrix(B);
        vector<double> flatC(static_cast<size_t>(n) * n, 0.0);

        int deviceCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (deviceCount == 0) {
            cerr << "No CUDA-compatible GPU found." << endl;
            return 1;
        }

        CUDA_CHECK(cudaSetDevice(0));

        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

        double* dA = nullptr;
        double* dB = nullptr;
        double* dC = nullptr;

        CUDA_CHECK(cudaMalloc(&dA, bytes));
        CUDA_CHECK(cudaMalloc(&dB, bytes));
        CUDA_CHECK(cudaMalloc(&dC, bytes));

        CUDA_CHECK(cudaMemcpy(dA, flatA.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, flatB.data(), bytes, cudaMemcpyHostToDevice));

        const vector<pair<int, int>> blockConfigs = {
            {4, 4},
            {8, 8},
            {16, 8},
            {8, 16},
            {16, 16},
            {32, 8},
            {8, 32},
            {32, 16},
            {16, 32},
            {32, 32}
        };

        float bestTimeMs = numeric_limits<float>::max();
        dim3 bestBlock(1, 1);
        dim3 bestGrid(1, 1);

        cout << "CUDA device: " << prop.name << '\n';
        cout << "Matrix size: " << n << "x" << n << '\n';
        cout << "\nTesting CUDA configurations:\n";
        cout << left << setw(14) << "Block"
             << setw(14) << "Grid"
             << setw(16) << "Threads/block"
             << setw(16) << "Time (ms)"
             << "Status" << '\n';
        cout << string(70, '-') << '\n';
        cout << fixed << setprecision(6);

        for (const auto& [bx, by] : blockConfigs) {
            if (bx * by > prop.maxThreadsPerBlock) {
                cout << setw(14) << (to_string(bx) + "x" + to_string(by))
                     << setw(14) << "-"
                     << setw(16) << (bx * by)
                     << setw(16) << "-"
                     << "Skipped: too many threads per block" << '\n';
                continue;
            }

            const dim3 blockSize(bx, by);
            const dim3 gridSize((n + blockSize.x - 1) / blockSize.x,
                                (n + blockSize.y - 1) / blockSize.y);

            CUDA_CHECK(cudaMemset(dC, 0, bytes));

            cudaEvent_t startEvent{};
            cudaEvent_t stopEvent{};
            CUDA_CHECK(cudaEventCreate(&startEvent));
            CUDA_CHECK(cudaEventCreate(&stopEvent));

            CUDA_CHECK(cudaEventRecord(startEvent));
            multiplyMatricesKernel<<<gridSize, blockSize>>>(dA, dB, dC, n);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(stopEvent));
            CUDA_CHECK(cudaEventSynchronize(stopEvent));

            float elapsedMs = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent));

            CUDA_CHECK(cudaEventDestroy(startEvent));
            CUDA_CHECK(cudaEventDestroy(stopEvent));

            cout << setw(14) << (to_string(bx) + "x" + to_string(by))
                 << setw(14) << (to_string(gridSize.x) + "x" + to_string(gridSize.y))
                 << setw(16) << (bx * by)
                 << setw(16) << elapsedMs
                 << "OK" << '\n';

            if (elapsedMs < bestTimeMs) {
                bestTimeMs = elapsedMs;
                bestBlock = blockSize;
                bestGrid = gridSize;
            }
        }

        if (bestTimeMs == numeric_limits<float>::max()) {
            throw runtime_error("No valid CUDA block configuration found.");
        }

        CUDA_CHECK(cudaMemset(dC, 0, bytes));
        multiplyMatricesKernel<<<bestGrid, bestBlock>>>(dA, dB, dC, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(flatC.data(), dC, bytes, cudaMemcpyDeviceToHost));

        cout << string(70, '-') << '\n';
        cout << "Best configuration: block " << bestBlock.x << "x" << bestBlock.y
             << ", grid " << bestGrid.x << "x" << bestGrid.y
             << ", time " << bestTimeMs << " ms" << '\n';

        cout << "\nFirst 5 elements of the first row of result matrix C: ";
        for (int j = 0; j < min(n, 5); ++j) {
            cout << flatC[j] << ' ';
        }
        cout << '\n';

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));

        return 0;
    } catch (const exception& ex) {
        cerr << ex.what() << endl;
        return 1;
    }
}
