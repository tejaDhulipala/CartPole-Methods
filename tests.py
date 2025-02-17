import torch
from torch import tensor
import numpy as np

a = torch.nn.Softmax(0)
b = torch.tensor(np.arange(25), dtype=torch.float32)
c = torch.tensor([10] * 25, dtype=torch.float32)
network = torch.nn.Sequential(
    torch.nn.Linear(1, 125),
    torch.nn.ReLU(),
    torch.nn.Linear(125, 2)
)

lr = 0.01
optim = torch.optim.Adam(network.parameters(), lr, amsgrad=True)
g_vals = np.linspace(8, 12, 100)

def range(v0, theta, g):
    return v0 ** 2 * np.sin(2 * theta) / g 

a = torch.arange(20).reshape((10, 2))
b = torch.zeros((10,), dtype=int)
print(a)
print(b)
print(a[torch.arange(a.size(0)),b])






