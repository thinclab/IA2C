import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from multiagent_particle_env.core import Agent
from belief_filter import BeliefFilter
from torch.distributions import Categorical
import numpy as np

class Actor(nn.Module):
    """Decentralized actor network for individual agents"""
    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, obs):
        logits = self.net(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy

class Critic(nn.Module):
    def __init__(self, local_obs_dim, global_obs_dim, total_action_dim):
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


class MAPPOAgent(Agent):
    """Individual agent with actor-critic components"""

    def __init__(self, agent_id, local_obs_dim, global_obs_dim, action_dim, total_action_dim, device, num_model=4, num_env=1):
        super().__init__()
        self.id = agent_id
        self.actor = Actor(local_obs_dim, action_dim)
        self.actor.to(device='cuda')
        self.critic = Critic(local_obs_dim, global_obs_dim, total_action_dim)
        self.critic.to(device='cuda')

        self.optimizer = optim.Adam([
            {'params': self.actor.parameters()},
            {'params': self.critic.parameters(), 'lr': 1e-3}
        ], lr=3e-4)
        self.device = device
        self.bf = BeliefFilter(num_model, action_dim, num_env)
        # Hyperparameters
        self.gamma = 0.95
        self.eps_clip = 0.2
        self.K_epochs = 10
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5

    def update(self, batch, is_central_obs):
        """PPO update with centralized critic"""
        # Convert batch data to tensors
        obs = torch.FloatTensor(batch['obs']).to(self.device)
        #global_states = torch.FloatTensor(batch['global_states']).to(self.device)
        actions = torch.FloatTensor(batch['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(batch['log_probs']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).to(self.device)
        dones = torch.FloatTensor(batch['dones']).to(self.device)
        all_actions = torch.FloatTensor(batch['all_actions']).to(self.device)
        print(obs.device)
        # Calculate advantages
        with torch.no_grad():
            values = self.critic(obs, is_central_obs).squeeze()
            returns = self._compute_returns(rewards, dones)
            advantages = returns - values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO optimization loop
        for _ in range(self.K_epochs):
            # Get new action distribution
            mean, log_prob, entropy= self.actor(obs)
            #dist = Normal(mean, std)

            # Calculate new log probabilities
            new_log_probs = log_prob.sum(-1)
            entropy = entropy.mean()

            # Probability ratio
            ratios = torch.exp(new_log_probs - old_log_probs)

            # Surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Actor loss
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

            # Critic loss
            current_values = self.critic(obs, is_central_obs).squeeze()
            critic_loss = (returns - current_values).pow(2).mean()

            # Total loss
            loss = actor_loss + 0.5 * critic_loss

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

    def _compute_returns(self, rewards, dones):
        """Calculate discounted returns"""
        returns = []
        R = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - d)
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32).to(self.device)