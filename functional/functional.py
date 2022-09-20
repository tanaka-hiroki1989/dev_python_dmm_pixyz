import torch.nn.functional as F
import torch.nn as nn

class functionalBase(nn.Module):
    def __init__(self):
        super().__init__()

class relu(functionalBase):
    def __init__(self):
        super().__init__()
    def forward(x):
        return F.relu(x)

