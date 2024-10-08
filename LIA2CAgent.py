import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from nets import EncoderDecoderNetwork
from nets import ActorCriticNetwork
import OpenOrg
import torch.nn.functional as F
import distributions 
from distributions import CategoricalPdType

class LIA2CAgent:
    def __init__(self, observation_size, action_size, latent_size, lr=1e-3):
        self.latent_size = latent_size

        self.encoder_decoder = EncoderDecoderNetwork(observation_size, action_size, latent_size)
        self.actor_critic = ActorCriticNetwork(observation_size, action_size, latent_size)

        self.optimizer_ed = optim.Adam(self.encoder_decoder.parameters(), lr=lr)
        self.optimizer_ac = optim.Adam(self.actor_critic.parameters(), lr=lr)

        self.gamma = 0.99  # Discount factor

    def select_action(self, observation):
        """Selects an action based on the current policy."""
        observation_tensor = torch.FloatTensor(observation).unsqueeze(0)
        action_probs= self.actor_critic.actor(observation_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob

    def update(self, trajectory):
        """
        Updates the networks based on the collected trajectory.
        trajectory: list of (observation, action, reward, next_observation, done)
        """
        observations = torch.FloatTensor([step[0] for step in trajectory])
        actions = torch.LongTensor([step[1] for step in trajectory]).unsqueeze(1)
        rewards = torch.FloatTensor([step[2] for step in trajectory]).unsqueeze(1)
        next_observations = torch.FloatTensor([step[3] for step in trajectory])
        dones = torch.FloatTensor([step[4] for step in trajectory]).unsqueeze(1)

        # Convert actions to one-hot vectors
        action_one_hot = torch.zeros(len(actions), len(OpenOrg.EMPLOYEE_ACTIONS))
        action_one_hot.scatter_(1, actions, 1)

        # Initial action distribution (uniform)
        action_dist = torch.ones(len(OpenOrg.EMPLOYEE_ACTIONS)) / len(OpenOrg.EMPLOYEE_ACTIONS)
        action_dist = action_dist.unsqueeze(0).repeat(len(observations), 1)

        # Forward pass through encoder-decoder
        z, next_obs_pred, updated_action_dist = self.encoder_decoder(observations, action_one_hot, action_dist)

        # Compute loss for encoder-decoder
        loss_ed = self.compute_encoder_decoder_loss(next_observations, next_obs_pred, updated_action_dist, action_dist)

        # Forward pass through actor-critic
        _, state_values = self.actor_critic(observations, z.detach())

        # Compute returns
        returns = self.compute_returns(rewards, dones)

        # Compute advantage
        advantages = returns - state_values

        # Actor loss
        action_probs = self.actor_critic.actor(observations)
        action_dist = torch.distributions.Categorical(action_probs)
        log_probs = action_dist.log_prob(actions.squeeze())
        actor_loss = -(log_probs * advantages.detach()).mean()

        # Critic loss
        critic_loss = advantages.pow(2).mean()

        # Total loss
        loss_ac = actor_loss + critic_loss

        # Backpropagation
        self.optimizer_ed.zero_grad()
        loss_ed.backward()
        self.optimizer_ed.step()

        self.optimizer_ac.zero_grad()
        loss_ac.backward()
        self.optimizer_ac.step()

    def compute_encoder_decoder_loss(self, next_observations, next_obs_pred, updated_action_dist, action_dist):
        """
        Computes the loss for the encoder-decoder network.
        """
        # Observation reconstruction loss
        obs_loss = nn.functional.mse_loss(next_obs_pred, next_observations)

        # Dirichlet posterior maximization (simplified for illustration)
        dirichlet_prior = torch.distributions.Dirichlet(action_dist.mean(dim=0))
        dirichlet_posterior = torch.distributions.Dirichlet(updated_action_dist.mean(dim=0))
        kl_divergence = torch.distributions.kl_divergence(dirichlet_posterior, dirichlet_prior).mean()

        # Total loss
        loss = obs_loss + kl_divergence

        return loss

    def compute_returns(self, rewards, dones):
        """Computes discounted returns."""
        returns = []
        R = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.stack(returns)
        return returns
