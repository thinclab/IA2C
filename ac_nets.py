import numpy as np
import torch
import torch.nn as nn


class EncoderDecoderNetwork(nn.Module):
    def __init__(self, observation_size, action_size, latent_size):
        super(EncoderDecoderNetwork, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(observation_size + action_size, 128),
            nn.ReLU(),
            nn.Linear(128, latent_size),
            nn.ReLU()
        )

        self.decoder_observation = nn.Sequential(
            nn.Linear(latent_size, 128),
            nn.ReLU(),
            nn.Linear(128, observation_size),
            nn.Softmax(dim=-1)
        )

        self.decoder_action_dist = nn.Sequential(
            nn.Linear(latent_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, observation, action, action_dist):
        # Concatenate observation and action
        x = torch.cat([observation, action], dim=-1)
        z = self.encoder(x)

        # Decode next observation and updated action distribution
        next_observation_pred = self.decoder_observation(z)
        updated_action_dist = self.decoder_action_dist(z)

        return z, next_observation_pred, updated_action_dist
    
class ActorCriticNetwork(nn.Module):
    def __init__(self, observation_size, action_size, latent_size):
        super(ActorCriticNetwork, self).__init__()
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(observation_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
            nn.Softmax(dim=-1)
        )

        # Critic network (takes latent embedding as input)
        self.critic = nn.Sequential(
            nn.Linear(observation_size + latent_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, observation, latent):
        # Actor uses only the observation
        action_probs = self.actor(observation)

        # Critic uses observation and latent embedding
        critic_input = torch.cat([observation, latent], dim=-1)
        state_value = self.critic(critic_input)

        return action_probs, state_value
