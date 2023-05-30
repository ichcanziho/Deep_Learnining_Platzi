import numpy as np
import torch

print(torch.__version__)

escalar = torch.randn(1)
vector = torch.zeros(1, 10)
matriz = torch.ones(2, 2)

print(escalar)
print(vector)
print(matriz)

t5 = torch.randn(5, 2, 3)

print(t5)

t2 = torch.tensor([[2, 2], [3, 3]])

print(t2)

print("*" * 64)

print(f"La shape de la matriz es {matriz.shape}")
print(f"La shape de t5 es {t5.shape}")

print(f"La shape de la matriz es {matriz.ndim}")
print(f"La shape de t5 es {t5.ndim}")

print(f"El tensor matriz tiene elementos de tipo: {matriz.dtype}")

matriz_float32 = torch.tensor([[3.1, 3.2], [3.3, 3.4]])
matriz_uint64 = torch.tensor([[3, 3], [3, 3]])

print(matriz_float32.dtype, matriz_uint64.dtype)

print((matriz_float32 + matriz_uint64).dtype)

print(matriz_uint64.to(torch.float32))

print(matriz_uint64.device)

print(torch.cuda.is_available())

if torch.cuda.is_available():
    matriz_uint64_cuda = matriz_uint64.to(torch.device("cuda"))

    print(matriz_uint64_cuda, matriz_uint64_cuda.type())
    print(matriz_uint64_cuda.to("cpu", torch.float32))

print("*"*64)

print(type(matriz.numpy()))

vector = np.ones(5)
vector_torch = torch.from_numpy(vector)
print(vector_torch, vector_torch.dtype)

# create a tensor of zeros with shape (3, 4)
zeros_tensor = torch.zeros((3, 4))

# create a tensor of ones with shape (3, 4)
ones_tensor = torch.ones((3, 4))

# create a tensor of random values with shape (2, 2)
random_tensor = torch.randn(4)

print(random_tensor, random_tensor.shape)

# add two tensors element-wise
added_tensor = zeros_tensor + ones_tensor

# subtract two tensors element-wise
subtracted_tensor = zeros_tensor - ones_tensor

# multiply two tensors element-wise
multiplied_tensor = zeros_tensor * ones_tensor

# divide two tensors element-wise
divided_tensor = random_tensor / ones_tensor

print(divided_tensor)

# create two matrices
matrix1 = torch.randn(2,3)
matrix2 = torch.randn(3,2)

print(f"matrix1 shape: {matrix1.shape}")
print(f"matrix2 shape: {matrix2.shape}")

# perform matrix multiplication
matx1x2 = torch.matmul(matrix1, matrix2)
print(matx1x2, matx1x2.shape)
