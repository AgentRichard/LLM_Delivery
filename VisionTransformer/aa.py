import torch

a = torch.randn(2, 3, 4, 6)
print(*a.shape[:-1])

a = a.reshape(*a.shape[:-1], -1, 2)
print(a.shape)

a = torch.view_as_complex(a)
print(a.shape)