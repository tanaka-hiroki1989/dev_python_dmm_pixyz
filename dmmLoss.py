import pixyz
from pixyz.losses import CrossEntropy,KullbackLeibler,IterativeLoss

class DMMLoss(pixyz.losses.loss):
    def __init__(self,emitter,transition,rnn,combiner) :
        super().__init__()
        self.emitter=emitter
        self.transition=transition
        self.rnn=rnn
        self.combiner=combiner
"""
def DMMLoss(emitter,transition,rnn,combiner):
        step_loss = CrossEntropy(combiner, emitter) \
        + KullbackLeibler(combiner, transition)
        _loss = IterativeLoss(step_loss, 
                    max_iter=t_max, 
                    series_var=["x", "h"], 
                    update_value={"z": "z_prev"})
        return  _loss.expectation(rnn).mean()
"""