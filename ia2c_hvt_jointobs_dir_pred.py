"""
IA2C in HVT domain — DIRECTION PREDICTOR (approach vs depart) baseline
with Replay Buffer (critic off-policy, actor on-policy) and **Soft Time Limit**.

Soft Time Limit:
- If the episode does not end naturally and the step count reaches
  MAX_STEPS_PER_EPISODE, we **force a terminal** and apply a small penalty
  to both agents. This prevents extremely long stalemates and stabilizes
  data collection / training dynamics.

Notes:
- Comments are in English as requested.
- Critic trains **only** from replay (off-policy); Actor trains on-policy per episode,
  and only after replay contains enough samples (warm-up).
"""

import os, random
from collections import deque
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from multiagent_particle_env.make_env import make_env
from neptune import Run
from API_token import project, api_token

# ===== Project deps (assumed available) =====
from ac_nets import ActorNetwork, CriticNetwork

# =========================
# Config
# =========================
USE_NEPTUNE = True
exp_id = 609
save_dir = f"/home/lzeng/Thinclab Code/HVT/IA2C/ExperimentList/{exp_id}/"
os.makedirs(save_dir, exist_ok=True)

CUDA = True
device = 'cuda' if (CUDA and torch.cuda.is_available()) else 'cpu'

# Seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if device == 'cuda':
    torch.cuda.manual_seed_all(SEED)

# RL hyper-params
step_r_fac = 1.0
LR_C = 0.003
LR_A = 0.001
BETA  = 0.005
GAMMA_INT = 0.96
GAMMA_DEF = 0.95
NUM_EPISODES = 15000

# Soft time-limit settings
USE_SOFT_TIME_LIMIT   = True
MAX_STEPS_PER_EPISODE = 5000     # force end if exceeded and not naturally done
TIME_LIMIT_PENALTY_INT = -0.2    # terminal shaping penalty for intruder
TIME_LIMIT_PENALTY_DEF = -0.2    # terminal shaping penalty for defender

# Env
scenario = 'eot/simple_hvt_1v1_random_orig'
env = make_env(scenario_name=scenario, logging=True, done=True)
env.world.adv_respawn_pos = 0.9

# Spaces
n_features      = 12
n_actor_actions = 5
n_dir           = 2
n_critic_actions= n_actor_actions * n_dir  # 10

# =========================
# Replay Buffer Config
# =========================
REPLAY_ON = True
BUFFER_SIZE = 100_000
BATCH_SIZE = 2048
CRITIC_START_LEARNING_STEPS = 20_000
UPDATES_PER_STEP = 1
STORE_ON_CPU = True

# On-policy update gates
USE_ONLINE_UPDATES = True
ONLINE_CRITIC = False
ONLINE_ACTOR  = True
ACTOR_ONLINE_START_STEPS = 30_000

# =========================
# Networks
# =========================
critic1 = CriticNetwork('crit1', n_features, n_critic_actions, LR_C, cuda=(device=='cuda'))
critic2 = CriticNetwork('crit2', n_features, n_critic_actions, LR_C, cuda=(device=='cuda'))
actor1  = ActorNetwork('act1', n_features, n_actor_actions, LR_A, BETA, cuda=(device=='cuda'))
actor2  = ActorNetwork('act2', n_features, n_actor_actions, LR_A, BETA, cuda=(device=='cuda'))

# =========================
# Direction Predictors (pretrained GRUs)
# =========================
class DirectionPredictorGRU(nn.Module):
    def __init__(self, hidden=64, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size=4, hidden_size=hidden, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden//2), nn.SiLU(), nn.Dropout(0.1), nn.Linear(hidden//2, 1))
    @torch.no_grad()
    def step(self, obs_t: torch.Tensor, hx=None):
        if obs_t.dim()==1: obs_t = obs_t.unsqueeze(0)
        x = obs_t.unsqueeze(0)
        h, hx_new = self.gru(x, hx)
        logit = self.head(h).squeeze()
        prob  = torch.sigmoid(logit)
        pred  = int((prob >= 0.5).item())
        return pred, float(prob.item()), hx_new

DIR_MODEL_A = './models_full_nomask/gru_taskA_o1_nomask.pt'   # o1 -> defender→intruder
DIR_MODEL_B = './models_full_nomask/gru_taskB_o2_nomask.pt'   # o2 -> intruder→HVT

dir_pred_intruder = DirectionPredictorGRU().to(device)
dir_pred_defender = DirectionPredictorGRU().to(device)
dir_pred_intruder.load_state_dict(torch.load(DIR_MODEL_A, map_location=device))
dir_pred_defender.load_state_dict(torch.load(DIR_MODEL_B, map_location=device))
dir_pred_intruder.eval(); dir_pred_defender.eval()

# =========================
# Helpers
# =========================
def direction_label_from_env(env) -> Tuple[int,int]:
    p_int = env.agents[0].state.p_pos.copy(); v_int = env.agents[0].state.p_vel.copy()
    p_def = env.agents[1].state.p_pos.copy(); v_def = env.agents[1].state.p_vel.copy()
    p_hvt = env.world.landmarks[0].state.p_pos.copy()
    v_hvt = getattr(env.world.landmarks[0].state, 'p_vel', None)
    if v_hvt is None: v_hvt = np.zeros_like(p_hvt)
    rv_def_intr = np.dot(p_def - p_int, v_def - v_int)
    dir_def_to_intr = 1 if rv_def_intr < 0.0 else 0
    rv_int_hvt = np.dot(p_int - p_hvt, v_int - v_hvt)
    dir_int_to_hvt = 1 if rv_int_hvt < 0.0 else 0
    return dir_def_to_intr, dir_int_to_hvt

def seen_intruder_by_defender(o2_like: torch.Tensor) -> bool:
    return bool(o2_like[6].item()!=0.0 or o2_like[7].item()!=0.0)

def seen_defender_by_intruder(o1_like: torch.Tensor) -> bool:
    return bool(o1_like[6].item()!=0.0 or o1_like[7].item()!=0.0)

# =========================
# Replay Buffer
# =========================
class ReplayBuffer:
    def __init__(self, capacity, device='cpu'):
        self.capacity = capacity
        self.device = device
        self.size = 0
        self.ptr = 0
        self._alloc()
    def _alloc(self):
        cap = self.capacity
        self.obs1      = torch.zeros((cap, n_features), dtype=torch.float32)
        self.obs2      = torch.zeros((cap, n_features), dtype=torch.float32)
        self.next_obs1 = torch.zeros((cap, n_features), dtype=torch.float32)
        self.next_obs2 = torch.zeros((cap, n_features), dtype=torch.float32)
        self.a1 = torch.zeros((cap, 1), dtype=torch.long)
        self.a2 = torch.zeros((cap, 1), dtype=torch.long)
        self.r1 = torch.zeros((cap, 1), dtype=torch.float32)
        self.r2 = torch.zeros((cap, 1), dtype=torch.float32)
        self.done = torch.zeros((cap, 1), dtype=torch.float32)
        self.eff_dir_now  = torch.zeros((cap, 2), dtype=torch.long)
        self.eff_dir_next = torch.zeros((cap, 2), dtype=torch.long)
        self.gt_dir_now   = torch.zeros((cap, 2), dtype=torch.long)
        self.logit1 = torch.zeros((cap, n_actor_actions), dtype=torch.float32)
        self.logit2 = torch.zeros((cap, n_actor_actions), dtype=torch.float32)
    def push(self, o1, o2, a1, a2, r1, r2, done, o1_next, o2_next,
             eff_now_0, eff_now_1, eff_next_0, eff_next_1, gt_now_0, gt_now_1,
             logit1=None, logit2=None):
        i = self.ptr
        self.obs1[i]      = o1.detach().cpu(); self.obs2[i]      = o2.detach().cpu()
        self.next_obs1[i] = o1_next.detach().cpu(); self.next_obs2[i] = o2_next.detach().cpu()
        self.a1[i,0] = int(a1); self.a2[i,0] = int(a2)
        self.r1[i,0] = float(r1); self.r2[i,0] = float(r2)
        self.done[i,0] = float(done)
        self.eff_dir_now[i,0]  = int(eff_now_0);  self.eff_dir_now[i,1]  = int(eff_now_1)
        self.eff_dir_next[i,0] = int(eff_next_0); self.eff_dir_next[i,1] = int(eff_next_1)
        self.gt_dir_now[i,0]   = int(gt_now_0);   self.gt_dir_now[i,1]   = int(gt_now_1)
        if logit1 is not None: self.logit1[i] = logit1.detach().cpu()
        if logit2 is not None: self.logit2[i] = logit2.detach().cpu()
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,))
        def to_dev(x):
            x = x[idx]
            return x.to(device) if not STORE_ON_CPU else x.to(device, non_blocking=True)
        return {
            'obs1': to_dev(self.obs1), 'obs2': to_dev(self.obs2),
            'next_obs1': to_dev(self.next_obs1), 'next_obs2': to_dev(self.next_obs2),
            'a1': to_dev(self.a1), 'a2': to_dev(self.a2),
            'r1': to_dev(self.r1), 'r2': to_dev(self.r2),
            'done': to_dev(self.done),
            'eff_next': to_dev(self.eff_dir_next), 'gt_now': to_dev(self.gt_dir_now),
        }

# =========================
# Critic updates from replay (off-policy)
# =========================
def critic_updates_from_replay(replay, updates=1):
    logs = {}
    for _ in range(updates):
        batch = replay.sample(BATCH_SIZE)
        obs1, obs2 = batch['obs1'], batch['obs2']
        nobs1, nobs2 = batch['next_obs1'], batch['next_obs2']
        a1b, a2b = batch['a1'], batch['a2']
        r1b, r2b = batch['r1'], batch['r2']
        db = batch['done']
        eff_next = batch['eff_next']
        gt_now = batch['gt_now']
        # intruder critic
        nja1 = (a1b.squeeze(-1) * n_dir + eff_next[:,1]).long()
        onehot_next1 = F.one_hot(nja1, num_classes=n_critic_actions).float()
        Qnext1 = (critic1.run_main(nobs1, grad=False) * onehot_next1).sum(-1, keepdims=True)
        target1 = r1b + GAMMA_INT * Qnext1 * (1.0 - db)
        jt1 = (a1b.squeeze(-1) * n_dir + gt_now[:,1]).long()
        critic1.batch_update(obs1, jt1, target1)
        # defender critic
        nja2 = (a2b.squeeze(-1) * n_dir + eff_next[:,0]).long()
        onehot_next2 = F.one_hot(nja2, num_classes=n_critic_actions).float()
        Qnext2 = (critic2.run_main(nobs2, grad=False) * onehot_next2).sum(-1, keepdims=True)
        target2 = r2b + GAMMA_DEF * Qnext2 * (1.0 - db)
        jt2 = (a2b.squeeze(-1) * n_dir + gt_now[:,0]).long()
        critic2.batch_update(obs2, jt2, target2)
        logs = {
            'intruder_critic_loss': critic1.losses[-1],
            'defender_critic_loss': critic2.losses[-1],
        }
    return logs

# =========================
# Neptune
# =========================
if USE_NEPTUNE:
    run = Run(project=project, api_token=api_token,
              tags=['direction-predictor','replay','critic-offpolicy','actor-onpolicy','soft-time-limit'],
              capture_hardware_metrics=False)

# =========================
# Training Loop
# =========================
replay = ReplayBuffer(BUFFER_SIZE, device='cpu' if STORE_ON_CPU else device)
done_count = [[], []]

def to_tensor_list(lst, dtype=torch.float32):
    return torch.from_numpy(np.array(lst)).to(device=device, dtype=dtype)

for ep in range(NUM_EPISODES):
    print(ep)
    # curriculum (unchanged)
    if sum(done_count[1]) > 35 and sum(done_count[0]) > 35:
        env.world.agents[0].size = max(env.world.agents[0].size - 0.01, 0.01)
        env.world.agents[1].size = max(env.world.agents[1].size - 0.01, 0.01)
        o1, o2 = env.reset(True); done_count = [[], []]
    elif sum(done_count[1]) > 55:
        env.world.agents[0].size = max(env.world.agents[0].size - 0.01, 0.01)
        env.world.agents[1].size = max(env.world.agents[1].size - 0.01, 0.01)
        o1, o2 = env.reset(True); done_count = [[], []]
    elif sum(done_count[0]) > 55:
        env.world.agents[0].size = min(env.world.agents[0].size + 0.01, 0.07)
        env.world.agents[1].size = min(env.world.agents[1].size + 0.01, 0.07)
        o1, o2 = env.reset(True); done_count = [[], []]
    else:
        o1, o2 = env.reset(False)
    o1 = torch.tensor(o1, dtype=torch.float32, device=device)
    o2 = torch.tensor(o2, dtype=torch.float32, device=device)

    # per-episode buffers (dynamic)
    obs_list, next_obs_list = [], []
    reward_list, done_list = [], []
    a1_list, a2_list, a1n_list, a2n_list = [], [], [], []
    eff_now_list, eff_next_list, gt_now_list = [], [], []

    # initial actions
    a1, logit1 = actor1.sample_action(o1, grad=True)
    a2, logit2 = actor2.sample_action(o2, grad=True)

    hx_dir1 = hx_dir2 = None
    step_idx = 0
    while True:
        # GT directions
        g_def_to_intr, g_int_to_hvt = direction_label_from_env(env)
        # predictor current
        pred_def_to_intr_now, _, hx_dir1 = dir_pred_intruder.step(o1[6:10], hx_dir1)
        pred_int_to_hvt_now,  _, hx_dir2 = dir_pred_defender.step(o2[8:], hx_dir2)
        # visibility
        v1_now = seen_defender_by_intruder(o1)
        v2_now = seen_intruder_by_defender(o2)
        eff_def_to_intr_now = g_def_to_intr if v1_now else pred_def_to_intr_now
        eff_int_to_hvt_now  = g_int_to_hvt  if v2_now else pred_int_to_hvt_now

        # env step
        (o1_, o2_), decomposed_r, done_flags, _ = env.step(
            np.array([np.eye(n_actor_actions)[a1], np.eye(n_actor_actions)[a2]])
        )
        true_done = bool(np.any(done_flags))
        r = [sum(decomposed_r[0][:3]) * step_r_fac, sum(decomposed_r[1])]

        # next obs/actions
        o1_ = torch.tensor(o1_, dtype=torch.float32, device=device)
        o2_ = torch.tensor(o2_, dtype=torch.float32, device=device)
        a1_, logit1_ = actor1.sample_action(o1_, grad=True)
        a2_, logit2_ = actor2.sample_action(o2_, grad=True)

        # predictor next
        pred_def_to_intr_next,  _, hx_dir1 = dir_pred_intruder.step(o1_[6:10], hx_dir1)
        pred_int_to_hvt_next,   _, hx_dir2 = dir_pred_defender.step(o2_[8:], hx_dir2)
        g_def_to_intr_next, g_int_to_hvt_next = direction_label_from_env(env)
        v1_next = seen_defender_by_intruder(o1_)
        v2_next = seen_intruder_by_defender(o2_)
        eff_def_to_intr_next = g_def_to_intr_next if v1_next else pred_def_to_intr_next
        eff_int_to_hvt_next  = g_int_to_hvt_next  if v2_next else pred_int_to_hvt_next

        # soft time limit
        step_idx += 1
        hit_time_limit = False
        if (not true_done) and USE_SOFT_TIME_LIMIT and (step_idx >= MAX_STEPS_PER_EPISODE):
            hit_time_limit = True
            true_done = True
            r[0] += TIME_LIMIT_PENALTY_INT
            r[1] += TIME_LIMIT_PENALTY_DEF
            done_flags = [False, False]

        # append
        obs_list.append( torch.stack([o1, o2], dim=0).detach().cpu().numpy() )
        next_obs_list.append( torch.stack([o1_, o2_], dim=0).detach().cpu().numpy() )
        reward_list.append( np.array(r, dtype=np.float32) )
        done_list.append( np.array([1.0 if true_done else 0.0], dtype=np.float32) )
        a1_list.append(int(a1.item())); a2_list.append(int(a2.item()))
        a1n_list.append(int(a1_.item())); a2n_list.append(int(a2_.item()))
        eff_now_list.append(np.array([eff_int_to_hvt_now, eff_def_to_intr_now], dtype=np.int64))
        eff_next_list.append(np.array([eff_int_to_hvt_next, eff_def_to_intr_next], dtype=np.int64))
        gt_now_list.append( np.array([g_int_to_hvt, g_def_to_intr], dtype=np.int64))

        # push to replay
        if REPLAY_ON:
            replay.push(
                o1, o2, int(a1.item()), int(a2.item()), float(r[0]), float(r[1]), float(true_done),
                o1_, o2_, int(eff_int_to_hvt_now), int(eff_def_to_intr_now), int(eff_int_to_hvt_next), int(eff_def_to_intr_next),
                int(g_int_to_hvt), int(g_def_to_intr), logit1, logit2
            )
            if replay.size > CRITIC_START_LEARNING_STEPS:
                logs = critic_updates_from_replay(replay, updates=UPDATES_PER_STEP)
                if USE_NEPTUNE:
                    run['replay/intruder_critic_loss'].append(logs['intruder_critic_loss'])
                    run['replay/defender_critic_loss'].append(logs['defender_critic_loss'])

        # end?
        if true_done:
            done_count[0].append(1 if done_flags[0] else 0)
            done_count[1].append(1 if done_flags[1] else 0)
            if len(done_count[0])>100: done_count[0].pop(0)
            if len(done_count[1])>100: done_count[1].pop(0)
            if USE_NEPTUNE:
                run['train/ep_length'].append(step_idx)
                run['train/time_limit_forced'].append(int(hit_time_limit))
            break

        # carry
        o1, o2 = o1_, o2_
        a1, a2 = a1_, a2_
        logit1, logit2 = logit1_, logit2_

    # ===== Convert to tensors for on-policy actor updates =====
    T = len(obs_list)
    obs_t       = to_tensor_list(obs_list).view(T,1,2,n_features)
    next_obs_t  = to_tensor_list(next_obs_list).view(T,1,2,n_features)
    reward_t    = to_tensor_list(reward_list).view(T,1,2)
    dones_t     = to_tensor_list(done_list).view(T,1,1)
    a1_t        = torch.tensor(np.array(a1_list), dtype=torch.long, device=device).view(T,1,1)
    a2_t        = torch.tensor(np.array(a2_list), dtype=torch.long, device=device).view(T,1,1)
    a1n_t       = torch.tensor(np.array(a1n_list), dtype=torch.long, device=device).view(T,1,1)
    a2n_t       = torch.tensor(np.array(a2n_list), dtype=torch.long, device=device).view(T,1,1)
    eff_now_t   = torch.tensor(np.array(eff_now_list), dtype=torch.long, device=device).view(T,1,2)
    eff_next_t  = torch.tensor(np.array(eff_next_list), dtype=torch.long, device=device).view(T,1,2)
    gt_now_t    = torch.tensor(np.array(gt_now_list), dtype=torch.long, device=device).view(T,1,2)

    if USE_ONLINE_UPDATES:
        # intruder
        nja1 = a1n_t.int().squeeze(-1) * n_dir + eff_next_t[:,:,1].int()
        onehot_n1 = F.one_hot(nja1.long(), num_classes=n_critic_actions).float()
        Qnext1_on = (critic1.run_main(next_obs_t[:,:,0,:]) * onehot_n1).sum(-1, keepdims=True)
        adv_tgt1 = reward_t[:,:,0].unsqueeze(-1) + GAMMA_INT * Qnext1_on * (1 - dones_t)
        ja1 = a1_t.int().squeeze(-1) * n_dir + eff_now_t[:,:,1].int()
        Q1 = critic1.run_main(obs_t[:,:,0,:], grad=False)
        onehot_j1 = F.one_hot(ja1.long(), num_classes=n_critic_actions).float()
        adv1 = adv_tgt1 - (Q1 * onehot_j1).sum(-1, keepdims=True)
        # defender
        nja2 = a2n_t.int().squeeze(-1) * n_dir + eff_next_t[:,:,0].int()
        onehot_n2 = F.one_hot(nja2.long(), num_classes=n_critic_actions).float()
        Qnext2_on = (critic2.run_main(next_obs_t[:,:,1,:]) * onehot_n2).sum(-1, keepdims=True)
        adv_tgt2 = reward_t[:,:,1].unsqueeze(-1) + GAMMA_DEF * Qnext2_on * (1 - dones_t)
        ja2 = a2_t.int().squeeze(-1) * n_dir + eff_now_t[:,:,0].int()
        Q2 = critic2.run_main(obs_t[:,:,1,:], grad=False)
        onehot_j2 = F.one_hot(ja2.long(), num_classes=n_critic_actions).float()
        adv2 = adv_tgt2 - (Q2 * onehot_j2).sum(-1, keepdims=True)
        # gate
        if ONLINE_ACTOR and replay.size >= ACTOR_ONLINE_START_STEPS:
            actor1.batch_update(obs_t[:,:,0,:], a1_t, adv1)
            actor2.batch_update(obs_t[:,:,1,:], a2_t, adv2)
            if USE_NEPTUNE:
                run['online/intruder_act_loss'].append(actor1.losses[-1])
                run['online/defender_act_loss'].append(actor2.losses[-1])
                if len(critic1.losses)>0: run['online/intruder_critic_loss'].append(critic1.losses[-1])
                if len(critic2.losses)>0: run['online/defender_critic_loss'].append(critic2.losses[-1])

    # periodic save
    if (ep % 100 == 20):
        torch.save(actor1.net.state_dict(), os.path.join(save_dir, f"last_act1_{exp_id}"))
        torch.save(actor2.net.state_dict(), os.path.join(save_dir, f"last_act2_{exp_id}"))

# Final save
torch.save(actor1.net.state_dict(), os.path.join(save_dir, f"last_act1_{exp_id}"))
torch.save(actor2.net.state_dict(), os.path.join(save_dir, f"last_act2_{exp_id}"))
