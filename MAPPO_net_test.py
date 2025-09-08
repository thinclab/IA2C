from multiagent_particle_env.make_env import make_env
import torch
import numpy as np
import torch.nn as nn, torch.optim as optim
from torch.distributions import Categorical

#from mappo_ac_net import MAPPOAgent


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_size=64):
        super().__init__()
        # shared body
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
        )
        # policy head
        self.policy = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, n_actions),
            nn.Softmax(dim=-1),
        )
        # value head
        self.value = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
    def forward(self, x):
        h = self.shared(x)
        return self.policy(h), self.value(h).squeeze(-1)

# 2) Rollout buffer
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.values = []
        self.rewards = []
        self.dones = []
    def clear(self):
        self.__init__()

# 3) PPO agent
class PPOAgent:
    def __init__(self, obs_dim, n_actions, **hp):
        self.gamma       = hp.get('gamma', 0.98)
        self.lam         = hp.get('gae_lambda', 0.95)
        self.eps_clip    = hp.get('eps_clip', 0.2)
        self.K_epochs    = hp.get('K_epochs', 4)
        self.lr          = hp.get('lr', 1e-4)
        self.net         = ActorCritic(obs_dim, n_actions)
        self.optimizer   = optim.Adam(self.net.parameters(), lr=self.lr)
        self.buffer      = RolloutBuffer()

    def select_action(self, state):
        state = torch.from_numpy(state).float()
        with torch.no_grad():
            probs, value = self.net(state)
        dist   = Categorical(probs)
        action = dist.sample()
        return (
            action.item(),
            dist.log_prob(action),
            value.item()
        )

    def compute_gae(self, next_value):
        # GAE-Lambda advantage
        advs = []
        gae  = 0
        values   = self.buffer.values + [next_value]
        rewards  = self.buffer.rewards
        dones    = self.buffer.dones
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae   = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advs.insert(0, gae)
        returns = [ad + v for ad, v in zip(advs, self.buffer.values)]
        return np.asarray(advs), np.asarray(returns)

    def update(self, next_value):
        # 1) compute advantages & returns
        advs, returns = self.compute_gae(next_value)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        advs_tensor    = torch.tensor(advs, dtype=torch.float32)
        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        # convert buffer lists to tensors
        states  = torch.tensor(self.buffer.states,  dtype=torch.float32)
        actions = torch.tensor(self.buffer.actions, dtype=torch.int64)
        old_logprobs = torch.tensor(self.buffer.logprobs, dtype=torch.float32)
        accum_actor_loss = 0
        accum_critic_loss = 0
        accum_entroopy = 0
        # 2) PPO epochs
        for _ in range(self.K_epochs):
            # new policy / value
            probs, values = self.net(states)
            dist = Categorical(probs)
            new_logprobs = dist.log_prob(actions)
            entropy      = dist.entropy().mean()

            # surrogate loss
            ratios = torch.exp(new_logprobs - old_logprobs)
            surr1  = ratios * advs_tensor
            surr2  = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advs_tensor
            actor_loss  = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values, returns_tensor)
            loss        = actor_loss + 0.5*critic_loss - 0.01*entropy

            # gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            accum_actor_loss += actor_loss.item()
            accum_critic_loss += critic_loss.item()
            accum_entroopy += entropy.item()
        # clear buffer
        self.buffer.clear()
        return {'actor_loss': accum_actor_loss/self.K_epochs,
                'critic_loss': accum_critic_loss/self.K_epochs,
                'entropy': accum_entroopy/self.K_epochs}





