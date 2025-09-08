"""
IA2C in HVT domain — uses DIRECTION PREDICTOR (approach vs depart)
instead of ACTION PREDICTOR for opponent modeling.

This version configures:
- Critic: OFF-POLICY via replay buffer only (no online critic updates).
- Actor: ON-POLICY (per-episode), but starts training ONLY after the replay
  buffer accumulates a configured number of transitions (warm-up).
- **No time-limit truncation**: the episode continues **until env returns done=True**.
  We remove the artificial step cap; if there is no done, we keep stepping.

Key design points:
- Critic joint space: 5 actions x 2 directions = 10.
- Joint index for both agents: joint = self_action * 2 + other_direction
- Direction labels from real coordinates via sign of r·v (no mask).
- Two GRU direction predictors (pretrained offline) consume o1/o2 and output
  approach(1)/depart(0). If an agent OBSERVES the other (o[6:8] != 0), we use
  the GT direction instead of prediction for that agent at that step.
"""

import os, time, sys, random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions import Categorical  # used by your ActorNetwork internally

# ==== project deps (assumed available in your repo) ====
from ac_nets import ActorNetwork, CriticNetwork
from belief_filter import BeliefFilter
from multiagent_particle_env.make_env import make_env
from multiagent_particle_env.scenarios.eot.simple_hvt_1v1_random import Scenario
from multiagent_particle_env.logger import Logger
from neptune import Run
from API_token import project, api_token

# =========================
# Config
# =========================
USE_NEPTUNE = True
exp_id = 610
save_dir = f"/home/lzeng/Thinclab Code/HVT/IA2C/ExperimentList/{exp_id}/"
os.makedirs(save_dir, exist_ok=True)

CUDA = True
device = "cuda" if (CUDA and torch.cuda.is_available()) else "cpu"

auto_seed = 42
random.seed(auto_seed)
np.random.seed(auto_seed)
torch.manual_seed(auto_seed)
if device == "cuda":
    torch.cuda.manual_seed_all(auto_seed)

step_r_fac = 1.0
LR_C = 0.003
LR_A = 0.001
BETA  = 0.005
GAMMA_INT = 0.96
GAMMA_DEF = 0.95
NUM_EPISODES       = 15000

scenario = 'eot/simple_hvt_1v1_random_orig'
envs = make_env(scenario_name=scenario, logging=True, done=True)
envs.world.adv_respawn_pos = 0.9
n_envs          = 1
n_features      = 12
n_actor_actions = 5
n_dir           = 2
n_critic_actions= n_actor_actions * n_dir  # 10

# =========================
# Replay Buffer Config
# =========================
REPLAY_ON = True
BUFFER_SIZE = 20000
BATCH_SIZE = 2048
# Critic starts learning (from replay) after this many transitions are in buffer
CRITIC_START_LEARNING_STEPS = 1000
UPDATES_PER_STEP = 1
STORE_ON_CPU = True            # Store buffer on CPU and move sampled batches to GPU

# =========================
# Online Update Config (fine-grained)
# =========================
# We keep online updates enabled to allow ON-POLICY actor training only.
USE_ONLINE_UPDATES = True
ONLINE_CRITIC = False          # IMPORTANT: critic will NOT update online
ONLINE_ACTOR  = True           # Actor trains on-policy per episode
# Actor warm-up: do not train actor on-policy until replay size >= threshold
ACTOR_ONLINE_START_STEPS = 1500

# =========================
# Nets
# =========================
critic1 = CriticNetwork("crit1", n_features, n_critic_actions, LR_C, cuda=(device=="cuda"))
critic2 = CriticNetwork("crit2", n_features, n_critic_actions, LR_C, cuda=(device=="cuda"))

actor1  = ActorNetwork("act1", n_features, n_actor_actions, LR_A, BETA, cuda=(device=="cuda"))
actor2  = ActorNetwork("act2", n_features, n_actor_actions, LR_A, BETA, cuda=(device=="cuda"))

# =========================
# Direction Predictors (pretrained)
# - A) input o1 -> predict defender→intruder
# - B) input o2 -> predict intruder→HVT
# =========================
class DirectionPredictorGRU(nn.Module):
    def __init__(self, obs_dim, hidden=64, num_layers=1):
        super().__init__()
        # Offline training used 4-dim slices (o1[6:10] / o2[8:]) -> input_size=4
        self.gru = nn.GRU(input_size=4, hidden_size=hidden,
                          num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden//2), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(hidden//2, 1)
        )
    @torch.no_grad()
    def step(self, obs_t: torch.Tensor, hx=None):
        """Single-step inference for a 1D observation slice."""
        if obs_t.dim()==1: obs_t = obs_t.unsqueeze(0)  # [1,D]
        x = obs_t.unsqueeze(0)                          # [1,1,D]
        h, hx_new = self.gru(x, hx)                    # [1,1,H]
        logit = self.head(h).squeeze()
        prob  = torch.sigmoid(logit)
        pred  = int((prob >= 0.5).item())
        return pred, float(prob.item()), hx_new

# Set your model checkpoints here
DIR_MODEL_A = "./models_full_nomask/gru_taskA_o1_nomask.pt"   # o1 -> defender→intruder
DIR_MODEL_B = "./models_full_nomask/gru_taskB_o2_nomask.pt"   # o2 -> intruder→HVT

dir_pred_intruder = DirectionPredictorGRU(n_features).to(device)  # for Agent1's critic other-dir
dir_pred_defender = DirectionPredictorGRU(n_features).to(device)  # for Agent2's critic other-dir
dir_pred_intruder.load_state_dict(torch.load(DIR_MODEL_A, map_location=device))
dir_pred_defender.load_state_dict(torch.load(DIR_MODEL_B, map_location=device))
dir_pred_intruder.eval()
dir_pred_defender.eval()

# =========================
# Helpers
# =========================
def direction_label_from_env(env) -> tuple[int,int]:
    """
    Returns (dir_def_to_intruder, dir_intruder_to_hvt) as ints in {0,1}.
    1=approach (r·v<0), 0=depart. Uses current true state in env.
    """
    p_int = env.agents[0].state.p_pos.copy()
    v_int = env.agents[0].state.p_vel.copy()
    p_def = env.agents[1].state.p_pos.copy()
    v_def = env.agents[1].state.p_vel.copy()
    p_hvt = env.world.landmarks[0].state.p_pos.copy()
    v_hvt = getattr(env.world.landmarks[0].state, 'p_vel', None)
    if v_hvt is None: v_hvt = np.zeros_like(p_hvt)

    rv_def_intr = np.dot(p_def - p_int, v_def - v_int)
    dir_def_to_intr = 1 if rv_def_intr < 0.0 else 0

    rv_int_hvt = np.dot(p_int - p_hvt, v_int - v_hvt)
    dir_int_to_hvt = 1 if rv_int_hvt < 0.0 else 0
    return dir_def_to_intr, dir_int_to_hvt

def seen_intruder_by_defender(o2_like: torch.Tensor) -> bool:
    """Defender sees intruder if o2[6:8] != 0."""
    return bool(o2_like[6].item()!=0.0 or o2_like[7].item()!=0.0)

def seen_defender_by_intruder(o1_like: torch.Tensor) -> bool:
    """Intruder sees defender if o1[6:8] != 0."""
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
        # Observations
        self.obs1      = torch.zeros((cap, n_features), dtype=torch.float32)
        self.obs2      = torch.zeros((cap, n_features), dtype=torch.float32)
        self.next_obs1 = torch.zeros((cap, n_features), dtype=torch.float32)
        self.next_obs2 = torch.zeros((cap, n_features), dtype=torch.float32)
        # Actions & rewards
        self.a1 = torch.zeros((cap, 1), dtype=torch.long)
        self.a2 = torch.zeros((cap, 1), dtype=torch.long)
        self.r1 = torch.zeros((cap, 1), dtype=torch.float32)
        self.r2 = torch.zeros((cap, 1), dtype=torch.float32)
        self.done = torch.zeros((cap, 1), dtype=torch.float32)  # 1 only if TRUE env done
        # Directions
        # last dim: [0]=intruder→HVT (for agent2 critic), [1]=defender→intruder (for agent1 critic)
        self.eff_dir_now  = torch.zeros((cap, 2), dtype=torch.long)
        self.eff_dir_next = torch.zeros((cap, 2), dtype=torch.long)
        self.gt_dir_now   = torch.zeros((cap, 2), dtype=torch.long)
        # Optional behavior logits (for importance sampling if needed)
        self.logit1 = torch.zeros((cap, n_actor_actions), dtype=torch.float32)
        self.logit2 = torch.zeros((cap, n_actor_actions), dtype=torch.float32)

    def push(self, o1, o2, a1, a2, r1, r2, done,
             o1_next, o2_next,
             eff_now_0, eff_now_1, eff_next_0, eff_next_1, gt_now_0, gt_now_1,
             logit1=None, logit2=None):
        i = self.ptr
        # Observations/actions/rewards/done
        self.obs1[i]      = o1.detach().cpu()
        self.obs2[i]      = o2.detach().cpu()
        self.next_obs1[i] = o1_next.detach().cpu()
        self.next_obs2[i] = o2_next.detach().cpu()
        self.a1[i,0] = int(a1)
        self.a2[i,0] = int(a2)
        self.r1[i,0] = float(r1)
        self.r2[i,0] = float(r2)
        self.done[i,0] = float(done)  # true terminal only
        # Directions
        self.eff_dir_now[i,0]  = int(eff_now_0)
        self.eff_dir_now[i,1]  = int(eff_now_1)
        self.eff_dir_next[i,0] = int(eff_next_0)
        self.eff_dir_next[i,1] = int(eff_next_1)
        self.gt_dir_now[i,0]   = int(gt_now_0)
        self.gt_dir_now[i,1]   = int(gt_now_1)
        # Logits
        if logit1 is not None:
            self.logit1[i] = logit1.detach().cpu()
        if logit2 is not None:
            self.logit2[i] = logit2.detach().cpu()

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,))
        def to_dev(x):
            x = x[idx]
            return x.to(device) if not STORE_ON_CPU else x.to(device, non_blocking=True)
        batch = {
            'obs1': to_dev(self.obs1), 'obs2': to_dev(self.obs2),
            'next_obs1': to_dev(self.next_obs1), 'next_obs2': to_dev(self.next_obs2),
            'a1': to_dev(self.a1), 'a2': to_dev(self.a2),
            'r1': to_dev(self.r1), 'r2': to_dev(self.r2),
            'done': to_dev(self.done),
            'eff_now': to_dev(self.eff_dir_now),
            'eff_next': to_dev(self.eff_dir_next),
            'gt_now': to_dev(self.gt_dir_now),
            'logit1': to_dev(self.logit1), 'logit2': to_dev(self.logit2),
        }
        return batch

# =========================
# Replay-based Updates (CRITIC ONLY)
# =========================
def critic_updates_from_replay(replay, updates=1):
    logs = {}
    for _ in range(updates):
        batch = replay.sample(BATCH_SIZE)

        obs1, obs2 = batch['obs1'], batch['obs2']
        nobs1, nobs2 = batch['next_obs1'], batch['next_obs2']
        a1b, a2b = batch['a1'], batch['a2']
        r1b, r2b = batch['r1'], batch['r2']
        db = batch['done']  # 1 only for TRUE terminal
        eff_next = batch['eff_next']
        gt_now = batch['gt_now']

        # ===== Critic 1 (intruder) =====
        nja1 = (a1b.squeeze(-1) * n_dir + eff_next[:,1]).long()  # next joint uses def->intruder for intruder
        ep_next_joint1 = F.one_hot(nja1, num_classes=n_critic_actions).float()
        Q_next1 = (critic1.run_main(nobs1, grad=False) * ep_next_joint1).sum(-1, keepdims=True)
        target1 = r1b + GAMMA_INT * Q_next1 * (1.0 - db)

        jt1 = (a1b.squeeze(-1) * n_dir + gt_now[:,1]).long()     # supervise with GT dir for stability
        critic1.batch_update(obs1, jt1, target1)

        # ===== Critic 2 (defender) =====
        nja2 = (a2b.squeeze(-1) * n_dir + eff_next[:,0]).long()  # next joint uses int->HVT for defender
        ep_next_joint2 = F.one_hot(nja2, num_classes=n_critic_actions).float()
        Q_next2 = (critic2.run_main(nobs2, grad=False) * ep_next_joint2).sum(-1, keepdims=True)
        target2 = r2b + GAMMA_DEF * Q_next2 * (1.0 - db)

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
              tags=['direction-predictor','no-mask','joint=action*dir','replay','critic-offpolicy','actor-onpolicy','actor-warmup','no-time-cap'],
              capture_hardware_metrics=False)

# =========================
# Training Loop
# =========================
reward_hist = []
ep = 0

done_count = [[], []]

replay = ReplayBuffer(BUFFER_SIZE, device='cpu' if STORE_ON_CPU else device)

def to_tensor_list(lst, dtype=torch.float32):
    return torch.tensor(np.array(lst), dtype=dtype, device=device)

while ep < NUM_EPISODES:
    print(ep)

    # Dynamic episode buffers (variable length until env done)
    obs_list            = []  # each: [2, n_features]
    next_obs_list       = []
    reward_list         = []  # each: [2]
    done_list           = []  # each: [1] (1 only at the final step)

    true_action_1_list      = []  # scalars
    true_action_2_list      = []
    true_next_action_1_list = []
    true_next_action_2_list = []

    eff_dir_now_list   = []   # each: [2]  [0]=int->HVT, [1]=def->intr
    eff_dir_next_list  = []
    gt_dir_now_list    = []

    # Reset env and get initial obs
    if sum(done_count[1]) > 35 and sum(done_count[0]) > 35:
        #envs.world.agents[0].size = max(envs.world.agents[0].size - 0.01, 0.01)
        #envs.world.agents[1].size = max(envs.world.agents[1].size - 0.01, 0.01)
        envs.world.adv_respawn_pos = min(envs.world.adv_respawn_pos + 0.05, 0.8)

        o1, o2 = envs.reset(True)
        done_count = [[], []]
    elif sum(done_count[1]) > 55:
        #envs.world.agents[0].size = max(envs.world.agents[0].size - 0.01, 0.01)
        #envs.world.agents[1].size = max(envs.world.agents[1].size - 0.01, 0.01)
        envs.world.adv_respawn_pos = max(envs.world.adv_respawn_pos - 0.05, 0.75)

        o1, o2 = envs.reset(True)
        done_count = [[], []]
    elif sum(done_count[0]) > 55:
        #.world.agents[0].size = min(envs.world.agents[0].size + 0.01, 0.07)
        #envs.world.agents[1].size = min(envs.world.agents[1].size + 0.01, 0.07)
        envs.world.adv_respawn_pos = min(envs.world.adv_respawn_pos + 0.05, 0.35)

        o1, o2 = envs.reset(True)
        done_count = [[], []]
    else:
        o1, o2 = envs.reset(False)
    o1 = torch.tensor(o1, dtype=torch.float32, device=device)
    o2 = torch.tensor(o2, dtype=torch.float32, device=device)

    # Initial actions
    a1, logit1 = actor1.sample_action(o1, grad=True)
    a2, logit2 = actor2.sample_action(o2, grad=True)

    # Predictor hidden states
    hx_dir1, hx_dir2 = None, None

    step_idx = 0
    while True:
        # Current GT directions from true state
        g_def_to_intr, g_int_to_hvt = direction_label_from_env(envs)

        # Predictor (always compute; may be overridden by visibility)
        pred_def_to_intr_now, _, hx_dir1 = dir_pred_intruder.step(o1[6:10], hx_dir1)
        pred_int_to_hvt_now,  _, hx_dir2 = dir_pred_defender.step(o2[8:], hx_dir2)

        # Visibility
        v1_now = seen_defender_by_intruder(o1)  # intruder sees defender?
        v2_now = seen_intruder_by_defender(o2)  # defender sees intruder?

        # Effective current directions: use GT if visible, else prediction
        eff_def_to_intr_now = g_def_to_intr if v1_now else pred_def_to_intr_now
        eff_int_to_hvt_now  = g_int_to_hvt  if v2_now else pred_int_to_hvt_now

        # Step environment
        (o1_, o2_), decomposed_r, done_flags, info = envs.step(
            np.array([np.eye(n_actor_actions)[a1], np.eye(n_actor_actions)[a2]])
        )
        true_done = bool(np.any(done_flags))

        decomposed_r[0][0] *= step_r_fac
        r = [sum(decomposed_r[0][:3]), sum(decomposed_r[1])]

        # Next obs & next actions
        o1_ = torch.tensor(o1_, dtype=torch.float32, device=device)
        o2_ = torch.tensor(o2_, dtype=torch.float32, device=device)
        a1_, logit1_ = actor1.sample_action(o1_, grad=True)
        a2_, logit2_ = actor2.sample_action(o2_, grad=True)

        # Predictor for next step
        pred_def_to_intr_next,  _, hx_dir1 = dir_pred_intruder.step(o1_[6:10], hx_dir1)
        pred_int_to_hvt_next,   _, hx_dir2 = dir_pred_defender.step(o2_[8:], hx_dir2)

        # Next-step GT and visibility
        g_def_to_intr_next, g_int_to_hvt_next = direction_label_from_env(envs)
        v1_next = seen_defender_by_intruder(o1_)  # intruder sees defender at t+1?
        v2_next = seen_intruder_by_defender(o2_)  # defender sees intruder at t+1?

        eff_def_to_intr_next = g_def_to_intr_next if v1_next else pred_def_to_intr_next
        eff_int_to_hvt_next  = g_int_to_hvt_next  if v2_next else pred_int_to_hvt_next

        # Append to dynamic buffers
        obs_list.append( torch.stack([o1, o2], dim=0).detach().cpu().numpy() )
        next_obs_list.append( torch.stack([o1_, o2_], dim=0).detach().cpu().numpy() )
        reward_list.append( np.array(r, dtype=np.float32) )
        done_list.append( np.array([1.0 if true_done else 0.0], dtype=np.float32) )

        true_action_1_list.append( int(a1.item()) )
        true_action_2_list.append( int(a2.item()) )
        true_next_action_1_list.append( int(a1_.item()) )
        true_next_action_2_list.append( int(a2_.item()) )

        eff_dir_now_list.append( np.array([eff_int_to_hvt_now, eff_def_to_intr_now], dtype=np.int64) )
        eff_dir_next_list.append( np.array([eff_int_to_hvt_next, eff_def_to_intr_next], dtype=np.int64) )
        gt_dir_now_list.append(  np.array([g_int_to_hvt, g_def_to_intr], dtype=np.int64) )

        # Write to replay buffer (done=true only on true env termination)
        if REPLAY_ON:
            replay.push(
                o1, o2,
                int(a1.item()), int(a2.item()),
                float(r[0]), float(r[1]),
                float(true_done),
                o1_, o2_,
                int(eff_int_to_hvt_now),  int(eff_def_to_intr_now),
                int(eff_int_to_hvt_next), int(eff_def_to_intr_next),
                int(g_int_to_hvt),        int(g_def_to_intr),
                logit1, logit2
            )
            # Critic replay updates after warm-up
            if replay.size > CRITIC_START_LEARNING_STEPS:
                logs = critic_updates_from_replay(replay, updates=UPDATES_PER_STEP)
                if USE_NEPTUNE and logs:
                    run['replay/intruder_critic_loss'].append(logs['intruder_critic_loss'])
                    run['replay/defender_critic_loss'].append(logs['defender_critic_loss'])

        # Episode end?
        if true_done:
            # curriculum bookkeeping
            done_count[0].append(1 if done_flags[0] else 0)
            done_count[1].append(1 if done_flags[1] else 0)
            if len(done_count[0]) > 100: done_count[0].pop(0)
            if len(done_count[1]) > 100: done_count[1].pop(0)
            break

        # Carry to next step
        o1, o2 = o1_, o2_
        a1, a2 = a1_, a2_
        logit1, logit2 = logit1_, logit2_
        step_idx += 1

    # ===================== Convert dynamic buffers to tensors =====================
    # Shapes: T x 1 x ... to stay compatible with your original code
    T = len(obs_list)
    obs_t       = to_tensor_list(obs_list)        .view(T, 1, 2, n_features)
    next_obs_t  = to_tensor_list(next_obs_list)   .view(T, 1, 2, n_features)
    reward_t    = to_tensor_list(reward_list)     .view(T, 1, 2)
    dones_t     = to_tensor_list(done_list)       .view(T, 1, 1)

    true_action_1_t      = torch.tensor(np.array(true_action_1_list), dtype=torch.long, device=device).view(T,1,1)
    true_action_2_t      = torch.tensor(np.array(true_action_2_list), dtype=torch.long, device=device).view(T,1,1)
    true_next_action_1_t = torch.tensor(np.array(true_next_action_1_list), dtype=torch.long, device=device).view(T,1,1)
    true_next_action_2_t = torch.tensor(np.array(true_next_action_2_list), dtype=torch.long, device=device).view(T,1,1)

    eff_dir_now_t  = torch.tensor(np.array(eff_dir_now_list), dtype=torch.long, device=device).view(T,1,2)
    eff_dir_next_t = torch.tensor(np.array(eff_dir_next_list), dtype=torch.long, device=device).view(T,1,2)
    gt_dir_now_t   = torch.tensor(np.array(gt_dir_now_list),  dtype=torch.long, device=device).view(T,1,2)

    # ===================== ON-POLICY Actor Updates (per-episode) =====================
    if USE_ONLINE_UPDATES and replay.size >= ACTOR_ONLINE_START_STEPS:
        # Intruder terms
        nja1 = true_next_action_1_t.int().squeeze(-1) * n_dir + eff_dir_next_t[:, :, 1].int()
        ep_next_joint1 = F.one_hot(nja1.long(), num_classes=n_critic_actions).float()
        Q_next1_on = (critic1.run_main(next_obs_t[:, :, 0, :]) * ep_next_joint1).sum(-1, keepdims=True)
        adv_target1 = reward_t[:, :, 0].unsqueeze(-1) + GAMMA_INT * Q_next1_on * (1 - dones_t)

        ja1 = true_action_1_t.int().squeeze(-1) * n_dir + eff_dir_now_t[:, :, 1].int()
        Q1 = critic1.run_main(obs_t[:, :, 0, :], grad=False)
        ep_joint1 = F.one_hot(ja1.long(), num_classes=n_critic_actions).float()
        adv1 = adv_target1 - (Q1 * ep_joint1).sum(-1, keepdims=True)

        # Defender terms
        nja2 = true_next_action_2_t.int().squeeze(-1) * n_dir + eff_dir_next_t[:, :, 0].int()
        ep_next_joint2 = F.one_hot(nja2.long(), num_classes=n_critic_actions).float()
        Q_next2_on = (critic2.run_main(next_obs_t[:, :, 1, :]) * ep_next_joint2).sum(-1, keepdims=True)
        adv_target2 = reward_t[:, :, 1].unsqueeze(-1) + GAMMA_DEF * Q_next2_on * (1 - dones_t)

        ja2 = true_action_2_t.int().squeeze(-1) * n_dir + eff_dir_now_t[:, :, 0].int()
        Q2 = critic2.run_main(obs_t[:, :, 1, :], grad=False)
        ep_joint2 = F.one_hot(ja2.long(), num_classes=n_critic_actions).float()
        adv2 = adv_target2 - (Q2 * ep_joint2).sum(-1, keepdims=True)

        # Actor ON-POLICY with warm-up; Critic remains off-policy only
        if ONLINE_ACTOR and replay.size >= ACTOR_ONLINE_START_STEPS:
            actor1.batch_update(obs_t[:, :, 0, :], true_action_1_t, adv1)
            actor2.batch_update(obs_t[:, :, 1, :], true_action_2_t, adv2)

        if USE_NEPTUNE:
            a1_loss = actor1.losses[-1] if len(actor1.losses)>0 else None
            a2_loss = actor2.losses[-1] if len(actor2.losses)>0 else None
            c1_loss = critic1.losses[-1] if len(critic1.losses)>0 else None
            c2_loss = critic2.losses[-1] if len(critic2.losses)>0 else None
            run['online/intruder_act_loss'].append(a1_loss)
            run['online/defender_act_loss'].append(a2_loss)
            run['online/intruder_critic_loss'].append(c1_loss)
            run['online/defender_critic_loss'].append(c2_loss)
            run['train/ep_length'].append(T)

    # Logging & checkpoints
    if (ep % 100 == 20):
        print('Current ep:', ep)
        if len(actor1.losses)>0: print('Intruder A loss:', actor1.losses[-1])
        if len(actor2.losses)>0: print('Defender A loss:', actor2.losses[-1])
        if len(critic1.losses)>0: print('Intruder C loss:', critic1.losses[-1])
        if len(critic2.losses)>0: print('Defender C loss:', critic2.losses[-1])
        torch.save(actor1.net.state_dict(), os.path.join(save_dir, f"last_act1_{exp_id}"))
        torch.save(actor2.net.state_dict(), os.path.join(save_dir, f"last_act2_{exp_id}"))

    ep += 1

# Final save
print(f"Saving last episode actors (episode {ep})")
torch.save(actor1.net.state_dict(), os.path.join(save_dir, f"last_act1_{exp_id}"))
torch.save(actor2.net.state_dict(), os.path.join(save_dir, f"last_act2_{exp_id}"))

