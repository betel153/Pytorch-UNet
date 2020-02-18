import numpy as np
import torch
import torch.nn as nn
from torch import optim

x = torch.randn(1, 1, 2, 3)
y = torch.randn(1, 1, 2, 3)
z = torch.randn(1, 1, 2, 3)

cat = torch.cat([x, y, z], dim=1)
test=1
