import torch
import numpy as np

# ----------tensor initialization
# direct from data
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# from numpy
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# from another tensors
x_ones = torch.ones_like(x_data)  # retain the properties of x_data (rows,columns...not values)
# print(f"Ones Tensor:\n{x_ones} \n") #f means format

x_rand = torch.rand_like(x_data, dtype=torch.float)  # override the datatype of x_data
# print(f"Random Tensor:\n{x_rand}\n")

# with random or constant values
shape = (2, 3,)  # mean 2x3 matrix
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

# print(f"Random Tensor:\n{rand_tensor} \n")
# print(f"Ones Tensor:\n{ones_tensor} \n")
# print(f"Zeros Tensor:\n{zeros_tensor} \n")

# -------------attributes
tensor = torch.rand(3, 4)

# print(f"Shape of tensor: {tensor.shape}")
# print(f"Datatype of tensor: {tensor.dtype}")
# print(f"Device tensor is stored on: {tensor.device}")

# ---------------operations
# assign column1 to 0
tensor = torch.ones(4, 4)
tensor[:, 1] = 0
# use gpu
if torch.cuda.is_available():
    tensor = tensor.to('cuda')
# print(tensor)
# print(f"Device tensor is stored on: {tensor.device}")

# concatenate
t1 = torch.cat([tensor, tensor, tensor], dim=1)  # given dimension=1
# print(t1)

# multiplying
# print(f"tensor.mul(tensor)\n {tensor.mul(tensor)}\n") #element-wise product z[i][j]=x[i][j]*y[i][j]
# print(f"tensor * tensor\n {tensor*tensor}\n") #simple
# print(f"tensor.matmul(tensor.T)\n {tensor.matmul(tensor.T)}\n") #matrix multiplication
# print(f"tensor @ tensor\n {tensor@tensor.T}\n")

# in-place operation
tensor.add_(5)  # use suffix to mean in-place, this will change the tensor
# print(tensor)

# -----------------bridge with numpy
# tensor to Numpy
t = torch.ones(5)
print(f"t:{t}")
n = t.numpy()
print(f"n:{n}")

# they share the same memory, so the change will affect both of them
t.add_(1)
print(f"t:{t}")
print(f"n:{n}")

#numpy to tensor
n=np.ones(5)
t=torch.from_numpy(n)

np.add(n,1,out=n)
print(f"t:{t}")
print(f"n:{n}")
