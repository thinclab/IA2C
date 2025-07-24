'''====================================================================================
Generic Actor-Critic Network classes with functions to build, train, and run the NNs.

Copyright (C) August, 2024  Bikramjit Banerjee

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

hidden_size = 20

class NeuralNet(nn.Module):
    def __init__(self, state_dim, action_dim, b_actor=False, dropout_probs: float = 0.1):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, action_dim)
        self.drop = torch.nn.Dropout(p=dropout_probs)
        self.b_actor = b_actor

    def forward(self, s):
        out = F.relu(self.l1(s))
        out = self.drop(out)
        out = F.relu(self.l2(out))
        out = self.drop(out)
        if self.b_actor:
            out = self.l3(out)
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
        self.critic_loss = np.inf
        
    def run_main(self, obs, grad=False): # (N_S X N_E X N_F) -->  (N_S X N_E X N_A)
        if not grad:
            with torch.no_grad():
                out = self.net(obs)
        else:
            out = self.net(obs)
        return out
    
    def batch_update(self, obs, act, target, action_distribution=False): # (N_S X N_E X N_F), (N_S X N_E X 1), (N_S X N_E X 1)
        self.optimizer.zero_grad()
        torch.autograd.set_detect_anomaly(True)
        Q = self.net.forward(obs) #(N_S X N_E X N_A)
        Q.requires_grad_(True)
        if not action_distribution:
            q_sel = F.one_hot(act.squeeze(-1).long(), num_classes=self.num_outs).float() # (N_S X N_E X N_A)
        else:
            q_sel = act # (N_S X N_E X N_A)
        dot_prd = (Q*q_sel).sum(-1, keepdims=True) # (N_S X N_E X 1)
        #print(target.requires_grad)
        #print(dot_prd.requires_grad)
        loss = self.loss(target, dot_prd)
        #old_params = [param.clone().detach() for param in self.net.parameters()]
        loss.backward()
        self.optimizer.step()
        '''for i, param in enumerate(self.net.parameters()):
            diff = (param.detach() - old_params[i]).abs().sum()
            print(f"Param {i} changed by: {diff.item()}")'''
        if self.cuda:
            get_loss = loss.cpu().data.numpy()
        else:
            get_loss = loss.detach().numpy()
        self.losses.append(get_loss)
        if len(self.losses)>100:
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
        self.entropy = np.inf
        self.actor_loss = np.inf

    def sample_action(self, obs, grad=False): # (N_S X N_E X N_F) --> (N_S X N_E X 1)
        if not grad:
            with torch.no_grad():
                logits=self.net(obs)
                probs = F.softmax(logits, -1)
        else:
            logits = self.net(obs)
            probs = F.softmax(logits, -1)
        dist = Categorical(probs=probs)
        act = dist.sample()
        return act, logits

    def action_distribution(self, obs, grad=False): # (N_S X N_E X N_F) --> (N_S X N_E X N_A)
        if not grad:
            with torch.no_grad():
                out = self.net(obs)
        else:
            out = self.net(obs)
        return out
        
    def batch_update(self, obs, act, adv, retain=False):  # (N_S X N_E X N_F), (N_S X N_E X 1), (N_S X N_E X 1)
        self.optimizer.zero_grad()
        logits = self.net.forward(obs)
        probs = F.softmax(logits, -1)
        dist = Categorical(probs=probs)
        neglogp = - dist.log_prob(act.squeeze(-1)).unsqueeze(-1)
        pg_loss = adv * neglogp
        entropy = dist.entropy().unsqueeze(-1)
        self.entropy = entropy.mean()
        loss = (pg_loss - self.beta*entropy).mean()
        loss.backward(retain_graph=retain)
        self.optimizer.step()
        if self.cuda:
            get_loss = loss.cpu().data.numpy()
        else:
            get_loss = loss.detach().numpy()
        self.losses.append(get_loss)
        if len(self.losses)>100:
            del self.losses[0]
        self.actor_loss = np.mean(self.losses)

    def deterministic_action(self,  obs, grad=False):
        if not grad:
            with torch.no_grad():
                probs=self.net(obs)
        else:
            probs=self.net(obs)
        #dist = Categorical(probs=probs)
        act = torch.argmax(probs, dim=-1)
        return act