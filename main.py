import argparse
from diCombiner import NormalCombiner
from emitter import BernoulliEmitter
from transition import NormalTransition
from rnn import RNN
from dmmLoss import DMMLoss
from pixyz.models import Model,VI
import torch
import torch.optim as optim
import tqdm

device="cpu"
batch_size = 128
epochs = 5
seed = 1
torch.manual_seed(seed)


def main(args):
    obs_dim = 88
    latent_dim=100
    rnn_dim=600
    t_max = obs_dim

    emitter = BernoulliEmitter().to(device)
    transition = NormalTransition().to(device)
    combiner = NormalCombiner().to(device)
    rnn = RNN().to(device)
  
    dmm_loss = DMMLoss(emitter=emitter,
        transition=transition,
        combiner=combiner,
        rnn=rnn)
    
    adam_params = {
        "lr": args.learning_rate,
        "betas": (args.beta1, args.beta2),
        "clip_norm": args.clip_norm,
        "lrd": args.lr_decay,
        "weight_decay": args.weight_decay,
    }

    dmm = Model(dmm_loss, distributions=[rnn, combiner, emitter, transition], 
                optimizer=optim.adam, 
                optimizer_params={"lr": 5e-4}, 
                clip_grad_value=10)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument("-n", "--num-epochs", type=int, default=5000)
    parser.add_argument("-lr", "--learning-rate", type=float, default=0.0003)
    parser.add_argument("-b1", "--beta1", type=float, default=0.96)
    parser.add_argument("-b2", "--beta2", type=float, default=0.999)
    parser.add_argument("-cn", "--clip-norm", type=float, default=10.0)
    parser.add_argument("-lrd", "--lr-decay", type=float, default=0.99996)
    parser.add_argument("-wd", "--weight-decay", type=float, default=2.0)
    parser.add_argument("-mbs", "--mini-batch-size", type=int, default=20)
    parser.add_argument("-ae", "--annealing-epochs", type=int, default=1000)
    parser.add_argument("-maf", "--minimum-annealing-factor", type=float, default=0.2)
    parser.add_argument("-rdr", "--rnn-dropout-rate", type=float, default=0.1)
    parser.add_argument("-iafs", "--num-iafs", type=int, default=0)
    parser.add_argument("-id", "--iaf-dim", type=int, default=100)
    parser.add_argument("-cf", "--checkpoint-freq", type=int, default=0)
    parser.add_argument("-lopt", "--load-opt", type=str, default="")
    parser.add_argument("-lmod", "--load-model", type=str, default="")
    parser.add_argument("-sopt", "--save-opt", type=str, default="")
    parser.add_argument("-smod", "--save-model", type=str, default="")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--jit", action="store_true")
    parser.add_argument("--tmc", action="store_true")
    parser.add_argument("--tmcelbo", action="store_true")
    parser.add_argument("--tmc-num-samples", default=10, type=int)
    parser.add_argument("-l", "--log", type=str, default="dmm.log")
    args = parser.parse_args()

    main(args)