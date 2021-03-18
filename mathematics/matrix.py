def get_matrix_average(lines, columns, matrix):
    sum_vector = [0, 0, 0]
    for i in range(lines):
        for j in range(columns):
            sum_vector[j] = sum_vector[j] + matrix[i][j]

    for i in range(columns):
        sum_vector[i] = sum_vector[i] / lines

    return sum_vector
