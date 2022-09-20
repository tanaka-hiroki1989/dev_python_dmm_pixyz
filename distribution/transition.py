from pixyz.distributions import Distribution, Normal

class Transition(Distribution):
    def __init__(self,layers,activations):
        super().__init__(cond_var=["z_prev"], var=["z"],name="p_trans")
        for name,layer in layers.items():
            setattr(self,name,layer)
        for name,activation in activations.items():
            setattr(self,name,layer)
    def forward(self,z_prev):
        pass

class NormalTransition(Normal):
    def __init__(self,layers,activations):
        super().__init__()       
    def forward(self, z_prev):
        """
        h = F.relu(self.fc1(z_prev))
        """
        def loc(self,z_prev):
            return z_prev
        def scale(self,z_prev):
            return z_prev 
        return {"loc": loc(z_prev), "scale": scale(z_prev)}