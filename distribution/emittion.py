from tkinter.messagebox import YESNO
import torch
import torch.nn as nn
import torch.functional as F
from pixyz.distributions import Distribution, Bernoulli, Normal
############
### emitter
############
class Emittion(Distribution):
    def __init__():
        super().__init__(cond_var=["z"], var=["x"],name="p_emittion")
    def forward(self, z):
        pass

class BernoulliEmittion(Bernoulli):
    def __init__(self,layers,activations):
        super().__init__()
        for name,hidden_layer in layers.items():
            setattr(self,name,hidden_layer)
        for name,activation in activations.items():
            setattr(self,name,activation)
     
    def forward(self, z):
        def probs(self,z):
            h=z
            for i in range():
                h = self.activate[i](self.layer[i](h))
            return h
        return {"probs": probs(z)}

class NormalEmittion(Emittion,Normal):
    def __init__(self,layers,activations):
        super().__init__()
        
    def forward(self, z):
        def loc(self,z):
            h = z
        
            return h
    
        def scale(self,z):
            h=z
            return h
            
        return {"loc": loc(z), "scale": scale(z)}
