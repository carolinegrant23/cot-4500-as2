import numpy as np
np.set_printoptions(precision=7, suppress=True, linewidth=100)

#1 check
x = [3.6, 3.8, 3.9]
y = [1.675, 1.436, 1.318]

def nevilles_method(x_points, y_points, interpolation):
    n = len(x_points)
    p = np.zeros((n, n))
    p[:, 0] = y_points
    for j in range(1, n):
        for i in range(n - j):
            p[i, j] = ((interpolation - x[i + j]) * p[i, j - 1] - (interpolation - x[i]) * p[i + 1, j - 1]) / (x_points[i] - x_points[i + j])
    return p[0, n - 1]

interpolation = 3.7
result = nevilles_method(x, y, interpolation)

print(result)

#2 check
x_value = [7.2, 7.4, 7.5, 7.6]
y_value = [23.5492, 25.3913, 26.8224, 27.4589]

coef = [] 
for i in range(len(x_value)): 
    coef.append(y_value[i]) 

for j in range(1, len(x_value)): 
    for i in range(len(x_value)-1, j-1, -1): 
        coef[i] = (coef[i]-coef[i-1])/(x_value[i]-x_value[i-j]) 

reduced_array = coef.pop(0)
print(coef)

#3 check
x_value = [7.2, 7.4, 7.5, 7.6]
y_value = [23.5492, 25.3913, 26.8224, 27.4589]

coef = [] 
for i in range(len(x_value)): 
    coef.append(y_value[i]) 

for j in range(1, len(x_value)): 
    for i in range(len(x_value)-1, j-1, -1): 
        coef[i] = (coef[i]-coef[i-1])/(x_value[i]-x_value[i-j]) 

answer = 0

for i in range(len(coef)): 
    num = coef[i] 
    for j in range(i): 
        num = num * (7.3 - x_value[j]) 
    answer += num

print(answer)

#4
def apply_div_dif(matrix: np.array):
    size = len(matrix)
    for i in range(2, size):
        for j in range(2, i + 2):
            if j >= len(matrix[i]) or matrix[i][j] != 0:
                continue
            left: float = matrix[i][j - 1]
            diagonal_left: float = matrix[i - 1][j - 1]
            numerator: float = (left - diagonal_left)
            denominator = matrix[i][0] - matrix[i - j + 1][0]
            operation = numerator / denominator
            matrix[i][j] = operation

    return matrix


def hermite_interpolation():
    x_points = [3.6, 3.8, 3.9]
    y_points = [1.675, 1.436, 1.318]
    slopes = [-1.195, -1.188, -1.182]
    num_of_points = len(x_points)
    matrix = np.zeros((2 * num_of_points, 2 * num_of_points))
    for i, x in enumerate(x_points):
        matrix[2 * i][0] = x
        matrix[2 * i + 1][0] = x

    for i, y in enumerate(y_points):
        matrix[2 * i][1] = y
        matrix[2 * i + 1][1] = y

    for i, slope in enumerate(slopes):
        matrix[2 * i + 1][2] = slope

    filled_matrix = apply_div_dif(matrix)
    print(filled_matrix)
hermite_interpolation()

#5 check
x_value = np.array([2, 5, 8, 10])
y_value = np.array([3, 5, 7, 9])

n = len(x_value)
matrix = np.zeros((n, n))
matrix[0, 0] = 1
matrix[n-1, n-1] = 1
for i in range(1, n-1):
    matrix[i, i-1] = x_value[i] - x_value[i-1]
    matrix[i, i] = 2 * (x_value[i+1] - x_value[i-1])
    matrix[i, i+1] = x_value[i+1] - x_value[i]

vector_b = np.zeros(n)
for i in range(1, n-1):
    vector_b[i] = 3 * (y_value[i+1] - y_value[i]) / (x_value[i+1] - x_value[i]) - \
           3 * (y_value[i] - y_value[i-1]) / (x_value[i] - x_value[i-1])

vector_x = np.linalg.solve(matrix, vector_b)

print(matrix)
print(vector_b)
print(vector_x)
