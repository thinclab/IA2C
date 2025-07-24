import sys, time
import numpy as np
import torch

from ac_nets import *
from belief_filter import BeliefFilter
from multiagent_particle_env.make_env import make_env
from multiagent_particle_env.scenarios.eot.simple_hvt_1v1_random import Scenario
from torch.utils.data import Dataset, DataLoader
from LIAM_Act_Predictor import LIAMActionPredictor
import random
import pandas as pd
import matplotlib.pyplot as plt

CUDA=True
NUM_EPISODES = 50
STEPS_PER_EPISODE = 50

scenario='eot/simple_hvt_1v1_random_orig'
envs=make_env(scenario_name=scenario, logging=True, done=True)
n_features = 12 #Joint observations for now
n_actor_actions = 5
loss1_cur = []
loss1_next = []
loss2_cur = []
loss2_next = []
device = 'cpu' if not CUDA else 'cuda'


actor1 = ActorNetwork("act1", n_features, n_actor_actions, 1.0, 1.0, cuda=CUDA)
actor2 = ActorNetwork("act2", n_features, n_actor_actions, 1.0, 1.0, cuda=CUDA)
act_predictor1 = LIAMActionPredictor(n_features, n_actor_actions)
act_predictor2 = LIAMActionPredictor(n_features, n_actor_actions)
act_predictor1.to(device='cuda')
act_predictor2.to(device='cuda')
#==============Load defender's best policy, and attacker's contemporary policy======================
actor1.net.load_state_dict(torch.load("/home/lzeng/Thinclab Code/HVT/IA2C/ExperimentList/394/last_act1_394"))
actor2.net.load_state_dict(torch.load("/home/lzeng/Thinclab Code/HVT/IA2C/ExperimentList/394/last_act2_394"))

o1_history, o2_history, a1_history, a2_history, next_a1_history, next_a2_history = [], [], [], [], [], []
buffer_sample_size = 64
buffer_size = 10000

def data_collection():
    data = []
    for ind in range(buffer_size):
        print(ind)
        o1, o2 = envs.reset()
        # print("HVT @",envs.world.landmarks[0].state.p_pos)
        ep_r = 0
        o1 = torch.tensor(o1, dtype=torch.float, device=device)
        o2 = torch.tensor(o2, dtype=torch.float, device=device)
        #a1, _ = actor1.sample_action(o1)
        #a2, _ = actor2.sample_action(o2)
        a1 = actor1.deterministic_action(o1)
        a2 = actor2.deterministic_action(o2)
        # print(envs.agents[0].state.p_pos)
        # print(envs.agents[1].state.p_pos)
        delta_pos = envs.agents[0].state.p_pos - envs.agents[1].state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))

        o1_history = []
        o2_history = []
        a1_history = []
        a2_history = []
        next_a1_history = []
        next_a2_history = []

        o1_history.append(o1.cpu())
        o2_history.append(o2.cpu())
        a1_history.append(a1.cpu())
        a2_history.append(a2.cpu())
        # print(dist)
        # print(envs.agents[0].size + envs.agents[1].size)
        # p_obs1, p_obs2 = noisy_private_obs(a1.detach(), a2.detach())
        # prior1, prior2 = bf1.prior, bf2.prior
        # _, prior1, pa2 = bf1.update(p_obs1, prior1) # Outputs predicted action of other agent
        # _, prior2, pa1 = bf2.update(p_obs2, prior2)
        ep_r = [0, 0]
        for step in range(STEPS_PER_EPISODE):
            (o1_, o2_), decomposed_r, done, info = envs.step(
                np.array([np.eye(n_actor_actions)[a1], np.eye(n_actor_actions)[a2]]))  # Step in environment
            # print(step, decomposed_r[0][1], decomposed_r[1][1], envs.agents[1].state.p_pos, envs.agents[0].state.p_pos)
            r = [sum(decomposed_r[0][:3]), sum(decomposed_r[1])]
            # print(o2_)

            # ===================================================================================================
            o1_ = torch.tensor(o1_, dtype=torch.float, device=device)
            o2_ = torch.tensor(o2_, dtype=torch.float, device=device)
            #a1_, _ = actor1.sample_action(o1_)
            #a2_, _ = actor2.sample_action(o2_)
            a1_ = actor1.deterministic_action(o1_)
            a2_ = actor2.deterministic_action(o2_)
            next_a1_history.append(a1_.cpu())
            next_a2_history.append(a2_.cpu())
            # print(r, ep_r)
            ep_r[0] += r[0]
            ep_r[1] += r[1]
            if np.any(done):
                # print(f'Done occurred in episode {ep}, step {step}: {done}. Rewards={ep_r}')
                break
            else:
                (o1, o2), (a1, a2) = (o1_, o2_), (a1_, a2_)
                o1_history.append(o1.cpu())
                o2_history.append(o2.cpu())
                a1_history.append(a1.cpu())
                a2_history.append(a2.cpu())
        data.append((np.stack(o1_history), np.stack(o2_history), np.array(a1_history), np.array(a2_history)))
    return data

class ExpertDataset(Dataset):
    def __init__(self, trajs):
        # trajs: list of (obs_seq [T,obs_dim], act_seq [T])
        self.trajs = trajs

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, idx):
        obs1, obs2, acts1, acts2 = self.trajs[idx]
        return (torch.tensor(obs1, dtype=torch.float32), torch.tensor(obs2, dtype=torch.float32),
                torch.tensor(acts1, dtype=torch.long), torch.tensor(acts2, dtype=torch.long))

def collate_fn(batch):
    # batch is list of (obs [T_i, D], acts [T_i])
    obs1_seqs, obs2_seq, act1_seqs, act2_seq = zip(*batch)
    lengths = [len(o) for o in obs1_seqs]
    # pad to max length
    maxL = max(lengths)
    D = obs1_seqs[0].size(1)
    padded_obs1 = torch.zeros(len(batch), maxL, D)
    padded_obs2 = torch.zeros(len(batch), maxL, D)

    padded_acts1 = torch.zeros(len(batch), maxL, dtype=torch.long)
    padded_acts2 = torch.zeros(len(batch), maxL, dtype=torch.long)

    mask = torch.zeros(len(batch), maxL, dtype=torch.bool)
    for i, (o1, o2, a1, a2) in enumerate(zip(obs1_seqs, obs2_seq, act1_seqs, act2_seq)):
        L = len(o1)
        padded_obs1[i, :L] = o1
        padded_acts1[i, :L] = a1

        padded_obs2[i, :L] = o2
        padded_acts2[i, :L] = a2
        mask[i, :L] = 1
    return padded_obs1, padded_obs2, padded_acts1, padded_acts2, mask

class PolicyImitatorRNNGRU(nn.Module):
    """
    GRU-based action predictor with ReLU non-linearities.
    """
    def __init__(self, obs_dim, n_actions, hidden_size=64, num_layers=1):
        super().__init__()
        # GRU backbone
        self.gru = nn.GRU(
            input_size=obs_dim + 1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        # Fully-connected classifier with ReLU
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x, y, lengths):
        """
        Args:
            x: Tensor of shape [batch, seq_len, obs_dim]
            lengths: list[int] of actual sequence lengths per batch
        Returns:
            logits: Tensor of shape [batch, seq_len, n_actions]
        """
        # Pack padded sequence for efficient GRU processing
        y = y.unsqueeze(-1)
        dataset = torch.cat((x, y), dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            dataset, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        # Unpack back to padded sequence
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        # Apply classifier to each time step
        logits = self.fc(out)  # [batch, seq_len, n_actions]
        return logits

def has_converged(loss_history, threshold=5e-3, window=10, relative=False):

    if len(loss_history) < window + 1:
        # Not enough history yet
        return False

    # take the last window+1 values to compute window deltas
    recent = loss_history[-(window + 1):]

    # compute deltas
    deltas = []
    for prev, curr in zip(recent, recent[1:]):
        delta = abs(curr - prev)
        if relative:
            delta = delta / (abs(prev) + 1e-8)
        deltas.append(delta)

    # check if all deltas are below threshold
    return all(d <= threshold for d in deltas)

def train_imitation(model1, model2, dataset, epochs=8000, lr=1e-3, batch_size=16):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    opt1 = torch.optim.Adam(model1.parameters(), lr=lr)
    opt2 = torch.optim.Adam(model2.parameters(), lr=lr)

    ce = nn.CrossEntropyLoss(reduction='none')
    for ep in range(epochs):
        total_loss1, total_loss2, total_tokens = 0, 0, 0
        for obs1, obs2, acts1, acts2, mask in loader:
            lengths = mask.sum(dim=1).tolist()
            logits1 = model1(obs1, acts1, lengths)  # [B, T, A]
            logits2 = model2(obs2, acts2, lengths)  # [B, T, A]

            B, T, A = logits1.shape
            loss1 = ce(logits1.view(B * T, A), acts2.view(-1))
            loss1 = (loss1 * mask.view(-1).float()).sum()
            total_loss1 += loss1.item()
            total_tokens += mask.sum().item()

            opt1.zero_grad()
            loss1.backward()
            opt1.step()

            loss2 = ce(logits2.view(B * T, A), acts1.view(-1))
            loss2 = (loss2 * mask.view(-1).float()).sum()
            total_loss2 += loss2.item()

            opt2.zero_grad()
            loss2.backward()
            opt2.step()

        print(f"Ep {ep + 1}/{epochs}, CE Loss1 per token = {total_loss1 / total_tokens:.4f}, CE Loss2 per token = {total_loss2 / total_tokens:.4f}")

def evaluate_imitation(model1: nn.Module,
                       model2: nn.Module,
                       dataset: torch.utils.data.Dataset,
                       batch_size: int = 32,
                       device: str = 'cpu'):
    """
    Runs the trained imitator over the dataset and prints overall accuracy.
    Assumes dataset.__getitem__ returns (obs_seq, act_seq) and that
    collate_fn pads and returns (obs, acts, mask).
    """
    model1.eval()
    model2.eval()

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=collate_fn)  # reuse your collate_fn

    total, correct1, correct2 = 0, 0, 0

    with torch.no_grad():
        for obs1, obs2, acts1, acts2, mask in loader:
            # obs: [B, T, D], acts: [B, T], mask: [B, T]
            obs1 = obs1.to(device)
            acts1 = acts1.to(device)
            obs2 = obs2.to(device)
            acts2 = acts2.to(device)
            mask = mask.to(device)

            lengths = mask.sum(dim=1).tolist()
            logits1 = model1(obs1, acts1, lengths)             # [B, T, A]
            preds1  = logits1.argmax(dim=-1)           # [B, T]
            logits2 = model2(obs2, acts2, lengths)             # [B, T, A]
            preds2  = logits2.argmax(dim=-1)
            # count only real timesteps
            valid = mask.bool()
            total  += valid.sum().item()
            correct1 += (preds1[valid] == acts2[valid]).sum().item()
            correct2 += (preds2[valid] == acts1[valid]).sum().item()

    acc1 = correct1 / total
    acc2 = correct2 / total

    print(f"Imitation accuracy over test set: {acc1*100:.2f}%")
    print(f"Imitation accuracy over test set: {acc2*100:.2f}%")

    return acc1, acc2

#Data Generation
lr = 1e-3

dataset = data_collection()
predictor1 = PolicyImitatorRNNGRU(n_features, n_actor_actions)
predictor2 = PolicyImitatorRNNGRU(n_features, n_actor_actions)

dataset = ExpertDataset(dataset)
train_imitation(predictor1, predictor2, dataset, batch_size=32, lr=5e-4)
print(evaluate_imitation(predictor1, predictor2, dataset))

"""for ep in range(NUM_EPISODES):
    print(ep)

    if len(o1_history) > buffer_sample_size:
        sample_batch_ind = random.sample(range(len(o1_history)), min(buffer_sample_size, len(o1_history)))
        # print(sample_batch_ind)
        o1_batch = []
        a1_batch = []
        next_a1_batch = []
        o2_batch = []
        a2_batch = []
        next_a2_batch = []
        for ind in sample_batch_ind:
            o1_batch.append(o1_history[ind])
            a1_batch.append(a1_history[ind])
            next_a1_batch.append(next_a1_history[ind])
            o2_batch.append(o2_history[ind])
            a2_batch.append(a2_history[ind])
            next_a2_batch.append(next_a2_history[ind])

        pad_o1_a1 = act_predictor1.padding(o1_batch, act_seq=a1_batch)
        pad_a1 = act_predictor1.padding(a1_batch)
        pad_next_a1 = act_predictor1.padding(next_a1_batch)
        pad_o2_a2 = act_predictor1.padding(o2_batch, act_seq=a2_batch)
        pad_a2 = act_predictor1.padding(a2_batch)
        pad_next_a2 = act_predictor1.padding(next_a2_batch)
        # o1_batch = torch.stack(o1_history[-1], dim=0)
        # a1_batch = torch.tensor(a1_history[-1], dtype=torch.float, device=device)
        # next_a1_batch = torch.tensor(next_a1_history[-1], dtype=torch.float, device=device)
        # o2_batch = torch.stack(o2_history[-1], dim=0)
        # a2_batch = torch.tensor(a2_history[-1], dtype=torch.float, device=device)
        # next_a2_batch = torch.tensor(next_a2_history[-1], dtype=torch.float, device=device)

        loss1 = act_predictor1.update(pad_o1_a1, pad_a2, pad_next_a2, buffer_sample_size)
        loss2 = act_predictor2.update(pad_o2_a2, pad_a1, pad_next_a1, buffer_sample_size)
        print(f"Intruder predictor current action loss: {loss1['loss_current']}")
        print(f"Intruder predictor next action loss: {loss1['loss_next']}")
        print(f"Defender predictor current action loss: {loss2['loss_current']}")
        print(f"Defender predictor current action loss: {loss2['loss_next']}")
        loss1_cur.append(loss1['loss_current'])
        loss1_next.append(loss1['loss_next'])
        loss2_cur.append(loss2['loss_current'])
        loss2_next.append(loss2['loss_next'])
        if has_converged(loss1_cur) and has_converged(loss2_cur) and has_converged(loss2_cur) and has_converged(loss2_cur):
            torch.save(act_predictor1.state_dict(), "intruder_act_predictor")
            torch.save(act_predictor2.state_dict(), "defender_act_predictor")
            break

torch.save(act_predictor1.state_dict(), "intruder_act_predictor")
torch.save(act_predictor2.state_dict(), "defender_act_predictor")
plt.figure()
plt.plot(loss1_cur)
plt.savefig("intruder_cur_act_loss")"""

plt.figure()
plt.plot(loss1_next)
plt.savefig("intruder_next_act_loss")

plt.figure()
plt.plot(loss2_cur)
plt.savefig("defender_cur_act_loss")

plt.figure()
plt.plot(loss2_next)
plt.savefig("defender_next_act_loss")