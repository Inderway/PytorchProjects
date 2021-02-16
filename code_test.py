import torch

# cumsum dim=0、-2, 为每列向下累加 dim=1、-1 为每行向下累加
# 对象为向量时，dim仅可等于0、-1， 为逐次向后累加
"""
print(torch.Tensor([[1, 2, 3],[4,5,6]]).cumsum(dim=0))
print(torch.Tensor([[1, 2, 3],[4,5,6]]).cumsum(dim=1))
print(torch.Tensor([[1, 2, 3],[4,5,6]]).cumsum(dim=-2))
print(torch.Tensor([[1, 2, 3],[4,5,6]]).cumsum(dim=-1))
print(torch.Tensor([1, 2, 3]).cumsum(dim=0))
print(torch.Tensor([1, 2, 3]).cumsum(dim=-1))
"""
for i in range(10):
    print('a')
print(i)
