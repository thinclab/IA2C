import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete

BETA = 0.001  # Exploration
LR_A = 0.000001  # (slower) learning rate for actor
LR_C = 0.00001


class CriticNetwork(nn.Module):
    def __init__(self, num_state_features, num_actions):
        super(CriticNetwork, self).__init__()
        self.loss_vec = []

        # Main Critic Network
        self.main_critic = nn.Sequential(
            nn.Linear(num_state_features, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, num_actions)
        )

        # Target Critic Network
        self.target_critic = nn.Sequential(
            nn.Linear(num_state_features, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, num_actions)
        )

        self.optimizer = optim.Adam(self.main_critic.parameters(), lr=LR_C)
        self.loss_fn = nn.MSELoss()

    def forward(self, state):
        return self.main_critic(state)

    def target_forward(self, state):
        return self.target_critic(state)

    def sync(self):
        self.target_critic.load_state_dict(self.main_critic.state_dict())

    def batch_update(self, x_vec, op_neuron_sel_vec, target_vec):
        num_steps = 10
        op_neuron_sel_vec = torch.tensor(op_neuron_sel_vec)
        target_vec = torch.tensor(target_vec).float()
        for _ in range(num_steps):
            q_values = self.forward(torch.tensor(x_vec).float())
            q_selected = q_values.gather(1, op_neuron_sel_vec.unsqueeze(-1)).squeeze(-1)
            loss = self.loss_fn(q_selected, target_vec)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.loss_vec.append(loss.item())
        self.value_loss = np.mean(self.loss_vec[-100:])

    def run_main(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        return self.forward(state).detach().numpy()[0]

    def run_target(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        return self.target_forward(state).detach().numpy()[0]


class ActorNetwork(nn.Module):
    def __init__(self, num_state_features, num_actions):
        super(ActorNetwork, self).__init__()
        

        self.actor = nn.Sequential(
            nn.Linear(num_state_features, 200),
            nn.Tanh(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, num_actions)
        )

        self.softmax = nn.Softmax(dim=-1)
        self.optimizer = optim.Adam(self.parameters(), lr=LR_A)

    def forward(self, state):
        return self.actor(state)

    def sample_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        logits = self.forward(state)
        probs = self.softmax(logits)
        action = torch.multinomial(probs, 1)
        return action.item()

    def best_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        logits = self.forward(state)
        return torch.argmax(logits, dim=-1).item()

    def action_distribution(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        logits = self.forward(state)
        probs = self.softmax(logits)
        return probs.detach().numpy()[0]

    def batch_update(self, x_vec, op_neuron_sel_vec, adv_vec):
        op_neuron_sel_vec = torch.tensor(op_neuron_sel_vec)
        adv_vec = torch.tensor(adv_vec)
        logits = self.forward(torch.tensor(x_vec).float())
        probs = self.softmax(logits)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(op_neuron_sel_vec)
        entropy = dist.entropy()
        loss = -(log_probs * adv_vec).mean() - BETA * entropy.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
