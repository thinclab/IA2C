from multiagent_particle_env.make_env import make_env
import torch
import numpy as np
import torch.nn as nn, torch.optim as optim
from torch.distributions import Categorical

#from mappo_ac_net import MAPPOAgent


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions, n_dir=2, hidden_size=64):
        super().__init__()
        self.n_actions = n_actions
        self.n_dir = n_dir
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
        # joint Q-head: Q(s, a, dir_other) with size n_actions * n_dir
        self.q_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, n_actions * n_dir),
        )

    def forward(self, x):
        h = self.shared(x)
        pi = self.policy(h)
        q_all = self.q_head(h)
        return pi, q_all

# 2) Rollout buffer
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.dirs = []  # direction labels of the other agent (0/1)
        self.logprobs = []
        self.values = []  # here: Q(s, a, dir_other) scalars for chosen joint
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
        self.n_actions   = n_actions
        self.n_dir       = hp.get('n_dir', 2)
        self.net         = ActorCritic(obs_dim, n_actions, n_dir=self.n_dir)
        self.optimizer   = optim.Adam(self.net.parameters(), lr=self.lr)
        self.buffer      = RolloutBuffer()

    def select_action(self, state, dir_other: int):
        state_t = torch.from_numpy(state).float()
        with torch.no_grad():
            probs, q_all = self.net(state_t)
        dist   = Categorical(probs)
        action = dist.sample()
        joint_idx = action.item() * self.n_dir + int(dir_other)
        q_flat = q_all.view(-1)
        q_scalar = q_flat[joint_idx]
        return (
            action.item(),
            dist.log_prob(action),
            q_scalar.item(),
        )

    def sample_action(self, state):
        state_t = torch.from_numpy(state).float()
        with torch.no_grad():
            probs, q_all = self.net(state_t)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item()

    def compute_gae(self, next_value):
        # GAE-Lambda advantage on Q(s,a,dir_other)
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

        advs_tensor    = torch.tensor(np.array(advs, dtype=np.float32))
        returns_tensor = torch.tensor(np.array(returns, dtype=np.float32))
        # convert buffer lists to tensors
        states  = torch.tensor(np.array(self.buffer.states,  dtype=np.float32))
        actions = torch.tensor(np.array(self.buffer.actions, dtype=np.int64))
        dirs    = torch.tensor(np.array(self.buffer.dirs,    dtype=np.int64))
        old_logprobs = torch.tensor(np.array(self.buffer.logprobs, dtype=np.float32))
        accum_actor_loss = 0.0
        accum_critic_loss = 0.0
        accum_entroopy = 0.0
        # 2) PPO epochs
        for _ in range(self.K_epochs):
            # new policy / joint-Q predictions
            probs, q_all = self.net(states)
            dist = Categorical(probs)
            new_logprobs = dist.log_prob(actions)
            entropy      = dist.entropy().mean()

            # gather Q(s, a, dir_other) for current actions/directions
            joint_idx = actions * self.n_dir + dirs
            q_pred = q_all.gather(1, joint_idx.view(-1, 1)).squeeze(-1)

            # surrogate loss
            ratios = torch.exp(new_logprobs - old_logprobs)
            surr1  = ratios * advs_tensor
            surr2  = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advs_tensor
            actor_loss  = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(q_pred, returns_tensor)
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
