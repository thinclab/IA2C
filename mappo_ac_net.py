import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.distributions import Categorical
import torch.nn.functional as F

class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, obs):
        logits = self.net(obs)
        return torch.distributions.Categorical(logits=logits)  # or Gaussian for continuous

class CriticNetwork(nn.Module):
    def __init__(self, local_obs_dim, global_obs_dim):
        super().__init__()

        self.local_critic = nn.Sequential(
            nn.Linear(local_obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.global_critic = nn.Sequential(
            nn.Linear(global_obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x, is_central_obs):
        if is_central_obs:
            return self.global_critic(x).squeeze(-1)
        else:
            return self.local_critic(x).squeeze(-1)
    
