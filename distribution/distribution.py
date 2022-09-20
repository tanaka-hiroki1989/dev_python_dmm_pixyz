from distutils.dist import Distribution
import torch
import torch.nn as nn
import torch.functional as F
from pixyz.distributions import Bernoulli, Normal

####################
# Transition (Prior)
####################



###########
# Combiner
###########
class NormalCombiner(Normal):
    def __init__(self,z_dim,h_dim):
        super(self).__init__(cond_var=["h", "z_prev"], var=["z"])
        self.fc1 = nn.Linear(z_dim, h_dim*2)
        self.fc21 = nn.Linear(h_dim*2, z_dim)
        self.fc22 = nn.Linear(h_dim*2, z_dim)
        
    def forward(self, h, z_prev):
        h_z = torch.tanh(self.fc1(z_prev))
        h = 0.5 * (h + h_z)
        return {"loc": self.fc21(h), "scale": F.softplus(self.fc22(h))}

