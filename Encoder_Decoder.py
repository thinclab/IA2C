import numpy as np
import torch
import torch.nn as nn
from numpy import dtype


class EncoderDecoderNetwork(nn.Module):
    def __init__(self, input_dim, latent_size, output_public_obs_dim, out_theta):
        super(EncoderDecoderNetwork, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_size),
        )

        self.decoder_observation = nn.Sequential(
            nn.Linear(latent_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_public_obs_dim)
        )

        self.decoder_action_dist = nn.Sequential(
            nn.Linear(latent_size, 128),
            nn.ReLU(),
            nn.Linear(128, out_theta),
            nn.Softmax(dim=-1)
        )

    def forward(self, public_obs, private_obs, action, theta):
        # Concatenate observation and action
        public_obs = torch.tensor(public_obs, dtype=torch.float32)  # Ensure proper dtype
        private_obs = torch.tensor(private_obs, dtype=torch.float32)
        action = torch.tensor([action], dtype=torch.float32)
        last_step_theta = torch.tensor(theta, dtype=torch.float32)
        x = torch.cat([public_obs,private_obs, action, last_step_theta], dim=-1)
        z = self.encoder(x)

        # Decode next observation and updated action distribution
        next_pub_obs = self.decoder_observation(z)
        next_private_obs = self.decoder_action_dist(z)

        return z, next_pub_obs, next_private_obs

    def batch_update(self):
        pass