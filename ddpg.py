# individual network settings for each actor + critic pair
# see networkforall for details

from networkforall import Network
from utilities import hard_update, gumbel_softmax, onehot_from_logits
from torch.optim import Adam
import torch
import numpy as np


# add OU noise for exploration
from OUNoise import OUNoise

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

class DDPGAgent:
    def __init__(self, in_actor, hidden_in_actor, hidden_out_actor, out_actor, lr_actor=1.0e-3):
        super(DDPGAgent, self).__init__()

        self.actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor=True).to(device)
        self.target_actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor=True).to(device)

        self.noise = OUNoise(out_actor, scale=1.0)

        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)

    def act(self, obs, noise=0.0):
        obs = obs.to(device)
        action = self.actor(obs) + noise*self.noise.noise()
        return action

    def target_act(self, obs, noise=0.0):
        obs = obs.to(device)
        action = self.target_actor(obs) + noise*self.noise.noise()
        return action
