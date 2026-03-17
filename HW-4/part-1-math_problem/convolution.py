import numpy as np
from scipy.signal import convolve2d

def matrix_with_max_pooling(matrix, stride=2):
    pool_size = 3
    rows, cols = matrix.shape

    # Calculate output dimensions
    out_rows = ((rows - pool_size) // stride) + 1
    out_cols = ((cols - pool_size) // stride) + 1

    # Create output matrix
    pooled = np.zeros((out_rows, out_cols), dtype=matrix.dtype)

    for i in range(out_rows):
        for j in range(out_cols):
            row_start = i * stride
            col_start = j * stride
            window = matrix[row_start:row_start+pool_size, col_start:col_start+pool_size]
            pooled[i, j] = np.max(window)

    return pooled
# Original 5x5 input matrix (before padding)
input_matrix = np.array([
    [0, 2, 4, 1, 0],
    [3, 1, 1, 0, 1],
    [2, 4, 1, 0, 1],
    [2, 0, 5, 2, 2],
    [0, 1, 3, 2, 1]
])

# 3x3 filter (edge detection)
filter_kernel = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

# Perform 2D convolution with 'same' padding and 'fill' boundary
matrix_stride_1 = convolve2d(input_matrix, filter_kernel, mode='same', boundary='fill', fillvalue=0)

# Display the result
print("\nConvolution Output (Stride = 1):")
print(matrix_stride_1)

print("\nMax Pooled: \n", matrix_with_max_pooling(matrix_stride_1))

stride = 2
matrix_stide_2 = matrix_stride_1[::stride, ::stride]
print("\n\nConvolution Output (Stride = 2):")
print(matrix_stide_2, "\n\n")



import numpy as np

# Input matrix with zero-padding (7x7 including padding)
input_matrix = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 2, 4, 1, 0, 0],
    [0, 3, 1, 1, 0, 1, 0],
    [0, 2, 4, 1, 0, 1, 0],
    [0, 2, 0, 5, 2, 2, 0],
    [0, 0, 1, 3, 2, 1, 0],
    [0, 0, 0, 0, 0, 0, 0],
])

# 3x3 filter (horizontal edge detection style)
filter_kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

# Convolution parameters
stride = 1
filter_size = filter_kernel.shape[0]
output_size = (input_matrix.shape[0] - filter_size) // stride + 1

# Output matrix
matrix_stride_1 = np.zeros((output_size, output_size), dtype=int)

# Convolution operation
for i in range(0, output_size):
    for j in range(0, output_size):
        region = input_matrix[i:i+filter_size, j:j+filter_size]
        matrix_stride_1[i, j] = np.sum(region * filter_kernel)

# Display result
print("Convolution Output (Stride = 1):")
print(matrix_stride_1)

