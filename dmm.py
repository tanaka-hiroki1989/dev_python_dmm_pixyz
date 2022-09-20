import pixyz
from pixyz.distributions import Deterministic
import torch
import torch.nn as nn

class RNN(Deterministic):
    def __init__(self,obs_dim,rnn_dim):
        super(self).__init__(cond_var=["x"], var=["h"])
        self.rnn = nn.GRU(
            input_size=obs_dim,
            hidden_size=rnn_dim, 
            bidirectional=False,
        )
        self.h0 = nn.Parameter(torch.zeros(2, 1, self.rnn.hidden_size))
        self.hidden_size = self.rnn.hidden_size
        
    def forward(self, x):
        h0 = self.h0.expand(2, x.size(1), self.rnn.hidden_size).contiguous()
        h, _ = self.rnn(x, h0)
        return {"h": h}

class DMM(pixyz.Model):
    def __init__(self,
        emitter,
        transition,
        combiner,
        rnn,
        num_iafs=0,
        iaf_dim=50,
        use_cuda=False):
        super().__init__(self)
        