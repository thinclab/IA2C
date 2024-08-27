'''====================================================================================
Generic Actor-Critic Network classes with functions to build, train, and run the NNs.

Copyright (C) May, 2024  Bikramjit Banerjee

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

===================================================================================='''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from torch.distributions import Categorical

hidden_size = 6

class NeuralNet(nn.Module):
    def __init__(self, state_dim, action_dim, b_actor=False):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, action_dim)
        self.b_actor = b_actor

    def forward(self, s):
        out = F.relu(self.l1(s))
        out = F.relu(self.l2(out))
        if self.b_actor:
            out = F.softmax(self.l3(out), -1)
        else:
            out = self.l3(out)
        return out
    
class CriticNetwork:
    def __init__(self, name, n_features, critic_actions, lr, cuda=False):
        self.num_outs = critic_actions
        self.net = NeuralNet(n_features, critic_actions)
        if cuda:
            self.net.cuda()
        self.loss = nn.MSELoss()
        self.optimizer = Adam(self.net.parameters(), lr=lr)
        self.cuda = cuda
        self.losses = []
        
    def run_main(self, obs, grad=False): # (N_S X N_E X N_F) -->  (N_S X N_E X N_A)
        if not grad:
            with torch.no_grad():
                out = self.net(obs)
        else:
            out = self.net(obs)
        return out
    
    def batch_update(self, obs, act, target, action_distribution=False): # (N_S X N_E X N_F), (N_S X N_E X 1), (N_S X N_E X 1)
        self.optimizer.zero_grad()
        Q = self.net.forward(obs) #(N_S X N_E X N_A)
        if not action_distribution:
            q_sel = F.one_hot(act.squeeze(-1).long(), num_classes=self.num_outs).float() # (N_S X N_E X N_A)
        else:
            q_sel = act # (N_S X N_E X N_A)
        dot_prd = (Q*q_sel).sum(-1, keepdims=True) # (N_S X N_E X 1)
        loss = self.loss(target, dot_prd)
        loss.backward()
        self.optimizer.step()
        if self.cuda:
            get_loss = loss.cpu().data.numpy()
        else:
            get_loss = loss.detach().numpy()
        self.losses.append(get_loss)
        if len(self.losses)>20:
            del self.losses[0]
        self.critic_loss = np.mean(self.losses)
        

class ActorNetwork:
    def __init__(self, name, n_features, actor_actions, lr, beta, cuda=False):
        self.num_outs = actor_actions
        self.net = NeuralNet(n_features, actor_actions, b_actor=True)
        if cuda:
            self.net.cuda()
        self.optimizer = Adam(self.net.parameters(), lr=lr)
        self.cuda = cuda
        self.beta = beta
        self.losses=[]

    def sample_action(self, obs, grad=False): # (N_S X N_E X N_F) --> (N_S X N_E X 1)
        if not grad:
            with torch.no_grad():
                probs=self.net(obs)
        else:
            probs=self.net(obs)
        dist = Categorical(probs=probs)
        act = dist.sample()
        return act

    def action_distribution(self, obs, grad=False): # (N_S X N_E X N_F) --> (N_S X N_E X N_A)
        if not grad:
            with torch.no_grad():
                out = self.net(obs)
        else:
            out = self.net(obs)
        return out
        
    def batch_update(self, obs, act, adv, retain=False):  # (N_S X N_E X N_F), (N_S X N_E X 1), (N_S X N_E X 1)
        dist = Categorical(probs=self.net.forward(obs))
        neglogp = - dist.log_prob(act.squeeze(-1)).unsqueeze(-1)
        pg_loss = adv * neglogp
        entropy = dist.entropy().unsqueeze(-1)
        loss = (pg_loss - self.beta*entropy).mean()
        loss.backward(retain_graph=retain)
        self.optimizer.step()
        if self.cuda:
            get_loss = loss.cpu().data.numpy()
        else:
            get_loss = loss.detach().numpy()
        self.losses.append(get_loss)
        if len(self.losses)>20:
            del self.losses[0]
        self.actor_loss = np.mean(self.losses)
