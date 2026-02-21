import numpy as np

def read_matrix(file):
    with open(file, encoding="utf-8") as f:
        n = int(f.readline().strip())
        data = [list(map(float, f.readline().split())) for _ in range(n)]
    return np.array(data)

A = read_matrix("matrixA.txt")
B = read_matrix("matrixB.txt")
C = read_matrix("result.txt")

C_true = A @ B

if np.allclose(C, C_true):
    print("Верификация пройдена: результат корректен")
else:
    print("Верификация НЕ пройдена: результат отличается от эталона")
    diff = np.max(np.abs(C - C_true))
    print("Максимальное абсолютное отклонение:", diff)

