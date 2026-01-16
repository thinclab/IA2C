import os
import random
from collections import deque
from typing import Tuple, List, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn

from multiagent_particle_env.make_env import make_env
from JointPPO_net import PPOAgent
import neptune
from API_token import api_token, mappo_project as mappo_project

# =========================
# Config
# =========================
scenario = 'eot/simple_hvt_1v1_random_orig'
env = make_env(scenario_name=scenario, logging=True, done=True)
USE_NEPTUNE = True
RUN_TAGS = ['onpolicy-rollout-buffer', 'soft-time-limit', 'difficulty-pools']

CUDA = True
DEVICE = 'cuda' if (CUDA and torch.cuda.is_available()) else 'cpu'

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE == 'cuda':
    torch.cuda.manual_seed_all(SEED)
# Direction Predictor paths
DIR_MODEL_INTRUDER = "./models_full_nomask/gru_taskA_o1_nomask_8.pt"
DIR_MODEL_DEFENDER = "./models_full_nomask/gru_taskB_o2_nomask_8.pt"
# PPO / env params
num_agents = 2
action_dim = 5
obs_dim = 12

N_DIR = 5                     # number of direction labels for joint Q (e.g., approach/depart)
NUM_UPDATES = 8000           # number of PPO update cycles (each after ROLLOUT_EPISODES episodes)
ROLLOUT_EPISODES = 16        # collect this many episodes before each PPO update

# Soft time-limit settings
USE_SOFT_TIME_LIMIT    = True
MAX_STEPS_PER_EPISODE  = 200     # force end if exceeded and not naturally done
c_step = 0.01
GAMMA_INTRUDER = 0.999
stall_mag = c_step * (1 - GAMMA_INTRUDER**MAX_STEPS_PER_EPISODE) / (1 - GAMMA_INTRUDER)  # ≈ 393.6 * c_step

k = 1.5  # 1.2~2.0
TIME_LIMIT_PENALTY_INT = - max(1.0, k * stall_mag)   # e.g., c_step=0.005 → -3.0
TIME_LIMIT_PENALTY_DEF = - max(0.5, 0.6 * k * stall_mag)

# logging / saving
exp_id = 177
model_path = f"./mappo_training_result/{exp_id}/"
os.makedirs(model_path, exist_ok=True)

# =========================
# Curriculum: Difficulty Pools
# =========================
DIFF_POOLS_INT: List[Dict[str, Any]] = [
    {"name": "i_easy",   "adv_respawn_pos": (0.55, 0.65)},
    {"name": "i_mid",    "adv_respawn_pos": (0.65, 0.75)},
    {"name": "i_hard",   "adv_respawn_pos": (0.7, 0.75)},
    {"name": "i_vhard",  "adv_respawn_pos": (0.75, 0.85)},
]
DIFF_POOLS_DEF: List[Dict[str, Any]] = [
    {"name": "d_easy",   "def_respawn_pos": (0.15, 0.25)},
    {"name": "d_mid",    "def_respawn_pos": (0.35, 0.45)},
    {"name": "d_hard",   "def_respawn_pos": (0.45, 0.55)},
    {"name": "d_vhard",  "def_respawn_pos": (0.55, 0.65)},
]

WINDOW = 100
TARGET_SR_INT = 0.45
TARGET_SR_DEF = 0.55
TARGET_LEN_INT = 180
TARGET_LEN_DEF = 180

GAIN_K_INT_SR  = 2.5
GAIN_K_DEF_SR  = 3.0
K_LEN_INT      = 1.2
K_LEN_DEF      = 0.0
LEN_BW_FRAC    = 0.25
USE_MAD_BW     = True

MIN_WEIGHT_EPS_INT = 0.05
MIN_WEIGHT_EPS_DEF = 0.05
HARDEST_MIN_SHARE_INT = 0.10
HARDEST_MIN_SHARE_DEF = 0.20

class PoolStats:
    def __init__(self, window: int):
        self.successes = deque(maxlen=window)
        self.episodes  = deque(maxlen=window)
        self.lengths   = deque(maxlen=window)  # 每集步数

POOL_STATS_INT = [PoolStats(WINDOW) for _ in DIFF_POOLS_INT]
POOL_STATS_DEF = [PoolStats(WINDOW) for _ in DIFF_POOLS_DEF]

def _sr(stats, neutral: float) -> float:
    if len(stats.episodes) == 0: return neutral
    return float(sum(stats.successes)) / max(len(stats.successes), 1)

def _len_median(stats, neutral: float) -> float:
    if len(stats.lengths) == 0: return neutral
    arr = np.asarray(stats.lengths, dtype=np.float32)
    return float(np.median(arr))

def _len_mad(stats, fallback_sigma: float) -> float:
    if len(stats.lengths) < 10:
        return fallback_sigma
    arr = np.asarray(stats.lengths, dtype=np.float32)
    med = np.median(arr)
    mad = np.median(np.abs(arr - med)) + 1e-6
    return float(1.4826 * mad)

def _apply_hardest_min_share(w: np.ndarray, min_share: float) -> np.ndarray:
    if min_share <= 0 or len(w) == 0: return w
    h = len(w) - 1
    total = w.sum()
    others = max(total - w[h], 1e-12)
    need_ratio = min_share / max(1.0 - min_share, 1e-12)
    if (w[h] / others) < need_ratio:
        w[h] = max(w[h], need_ratio * others)
    return w

def _length_bell_weight(len_med: float, target_len: float, k_len: float, sigma: float) -> float:
    z = (len_med - target_len) / max(sigma, 1e-6)
    return float(np.exp(-k_len * (z ** 2)))

def compute_weights_int(step: int = None, max_step: int = None):
    w = []
    for pid, st in enumerate(POOL_STATS_INT):
        sr  = _sr(st, TARGET_SR_INT)
        lm  = _len_median(st, TARGET_LEN_INT)
        w_sr = np.exp(GAIN_K_INT_SR * (TARGET_SR_INT - sr))
        sigma_fallback = LEN_BW_FRAC * TARGET_LEN_INT
        sigma = _len_mad(st, sigma_fallback) if USE_MAD_BW else sigma_fallback
        w_len = _length_bell_weight(lm, TARGET_LEN_INT, K_LEN_INT, sigma)
        w.append(w_sr * w_len)

    w = np.asarray(w, dtype=np.float64)
    w = np.maximum(w, MIN_WEIGHT_EPS_INT)

    # ====== hardest pool bias ======
    if step is not None and max_step is not None and max_step > 0:
        prog = np.clip(step / max_step, 0.0, 1.0)    # 0 → 开始, 1 → 结束
        # bias_factor from 1 to 1 + BIAS_MAX
        BIAS_MAX = 2.0
        bias_factor = 1.0 + BIAS_MAX * prog
        w[-1] *= bias_factor

    base_min_share = HARDEST_MIN_SHARE_INT  # 你现在是 0.10
    if step is not None and max_step is not None and max_step > 0:
        extra_share = 0.25 * np.clip(step / max_step, 0.0, 1.0)   # 最多再加 0.25
        local_min_share = min(base_min_share + extra_share, 0.5)  # 上限 0.5
    else:
        local_min_share = base_min_share

    w = _apply_hardest_min_share(w, local_min_share)
    w /= w.sum()
    return w

def global_defender_sr():
    all_s = 0; n = 0
    for st in POOL_STATS_DEF:
        all_s += sum(st.successes); n += max(len(st.successes), 1)
    return (all_s / n) if n > 0 else TARGET_SR_DEF

def compute_weights_def(step: int = None, max_step: int = None):
    gsr = global_defender_sr()
    if gsr < 0.35:
        # defender too weak, increase sr sampling weight
        k_sr = max(GAIN_K_DEF_SR, 3.5)
        k_len = 0.3
        hardest_share_base = max(HARDEST_MIN_SHARE_DEF, 0.20)
    elif gsr < 0.55:

        k_sr = GAIN_K_DEF_SR
        k_len = max(0.5 * K_LEN_DEF, 0.4)
        hardest_share_base = HARDEST_MIN_SHARE_DEF
    else:
        # defender too strong, increase episode length sampling weight
        k_sr = max(0.8 * GAIN_K_DEF_SR, 2.0)
        k_len = max(K_LEN_DEF, 1.0)
        hardest_share_base = HARDEST_MIN_SHARE_DEF

    w = []
    for pid, st in enumerate(POOL_STATS_DEF):
        sr = _sr(st, TARGET_SR_DEF)
        lm = _len_median(st, TARGET_LEN_DEF)

        w_sr = np.exp(k_sr * (TARGET_SR_DEF - sr))

        sigma_fb = LEN_BW_FRAC * TARGET_LEN_DEF
        sigma = _len_mad(st, sigma_fb) if USE_MAD_BW else sigma_fb
        w_len = _length_bell_weight(lm, TARGET_LEN_DEF, k_len, sigma) if k_len > 0 else 1.0

        w.append(w_sr * w_len)

    w = np.asarray(w, dtype=np.float64)
    w = np.maximum(w, MIN_WEIGHT_EPS_DEF)

    # ===== curriculum bias =====
    if step is not None and max_step is not None and max_step > 0:
        prog = np.clip(step / max_step, 0.0, 1.0)
        BIAS_MAX_DEF = 1.0  # defender 的 bias 小一点
        bias_factor = 1.0 + BIAS_MAX_DEF * prog
        w[-1] *= bias_factor  # 对 d_vhard 动手

    # ===== hardest_min_share  =====
    if step is not None and max_step is not None and max_step > 0:
        extra_share = 0.15 * np.clip(step / max_step, 0.0, 1.0)  # defender 就少抬一点
        local_min_share = min(hardest_share_base + extra_share, 0.5)
    else:
        local_min_share = hardest_share_base

    w = _apply_hardest_min_share(w, local_min_share)
    w /= w.sum()
    return w

def _sample_param(val):
    if isinstance(val, (tuple, list)):
        lo, hi = float(val[0]), float(val[1])
        return random.uniform(lo, hi)
    return val

def apply_dual_pool_params(env, i_pid: int, d_pid: int) -> Dict[str, float]:
    ip = DIFF_POOLS_INT[i_pid]
    dp = DIFF_POOLS_DEF[d_pid]
    sampled = {}
    if "adv_respawn_pos" in ip:
        v = _sample_param(ip["adv_respawn_pos"])
        env.world.adv_respawn_pos = v
        sampled['adv_respawn_pos'] = v
    if "def_respawn_pos" in dp:
        v = _sample_param(dp["def_respawn_pos"])
        env.world.def_respawn_pos = v
        sampled['def_respawn_pos'] = v
    return sampled

# =========================
# Agents
# =========================
intruder = PPOAgent(obs_dim-2, action_dim, gamma=0.99, n_dir=N_DIR)
defender = PPOAgent(obs_dim, action_dim, gamma=1.0, n_dir=N_DIR)

# =========================
# Utils
# =========================
class DirectionPredictorGRU(nn.Module):
    def __init__(self, obs_dim, hidden=64, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size=8, hidden_size=hidden,
                          num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden//2), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(hidden//2, 1)
        )
    @torch.no_grad()
    def step(self, obs_t: torch.Tensor, hx=None):
        """
        obs_t: Tensor[4] or [1,4]
        Returns: (pred_label:int{0,1}, prob:float in [0,1], new_hidden)
        """
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)  # [1,4]
        x = obs_t.unsqueeze(0)          # [1,1,4]
        h, hx_new = self.gru(x, hx)     # [1,1,H]
        logit = self.head(h).squeeze()
        prob  = torch.sigmoid(logit)
        pred  = int((prob >= 0.5).item())
        return pred, float(prob.item()), hx_new

dir_pred_intruder = DirectionPredictorGRU(obs_dim).to(DEVICE)  # used by intruder's critic (other = defender)
dir_pred_defender = DirectionPredictorGRU(obs_dim).to(DEVICE)  # used by defender's critic (other = intruder)
dir_pred_intruder.eval()
dir_pred_defender.eval()

def seen_intruder_by_defender(o2_like: torch.Tensor) -> bool:
    return bool((o2_like[6].item() != 0.0) or (o2_like[7].item() != 0.0))

def seen_defender_by_intruder(o1_like: torch.Tensor) -> bool:
    return bool((o1_like[6].item() != 0.0) or (o1_like[7].item() != 0.0))


def compute_directions_from_env(env) -> Tuple[int, int]:
    """
    Compute coarse direction labels from the true env state.
    dir_def_to_intruder: defender→intruder (1=approach, 0=depart)
    dir_intr_to_hvt: intruder→HVT (1=approach, 0=depart)
    NOTE: this uses ground-truth positions/velocities. If you want to
    use your pretrained direction predictors instead, replace the body
    of this function with predictor calls that return the same 0/1 labels.
    """
    p_int = env.agents[0].state.p_pos.copy()
    v_int = env.agents[0].state.p_vel.copy()
    p_def = env.agents[1].state.p_pos.copy()
    v_def = env.agents[1].state.p_vel.copy()
    p_hvt = env.world.landmarks[0].state.p_pos.copy()
    v_hvt = getattr(env.world.landmarks[0].state, 'p_vel', None)
    if v_hvt is None:
        v_hvt = np.zeros_like(p_hvt)

    rv_def_intr = np.dot(p_def - p_int, v_def - v_int)
    dir_def_to_intr = 1 if rv_def_intr < 0.0 else 0

    rv_int_hvt = np.dot(p_int - p_hvt, v_int - v_hvt)
    dir_int_to_hvt = 1 if rv_int_hvt < 0.0 else 0
    return int(dir_def_to_intr), int(dir_int_to_hvt)

def insert_data(agent: PPOAgent, obs: np.ndarray, act: int, dir_other: int,
                reward: float, logp: float, val: float, done: bool):
    agent.buffer.states.append(obs)
    agent.buffer.actions.append(act)
    agent.buffer.dirs.append(int(dir_other))
    agent.buffer.logprobs.append(logp)
    agent.buffer.values.append(val)
    agent.buffer.rewards.append(reward)
    agent.buffer.dones.append(1.0 if done else 0.0)

@torch.no_grad()
def select_action(agent: PPOAgent, obs: np.ndarray, dir_other: int) -> Tuple[int, float, float]:
    """Returns (action, logprob, Q_scalar for that (s,a,dir_other))."""
    a, logp, v = agent.select_action(obs, dir_other)
    return int(a), float(logp), float(v)

# =========================
# Neptune
# =========================
run = None
if USE_NEPTUNE:
    run = neptune.init_run(project=mappo_project, api_token=api_token, tags=RUN_TAGS)

def maybe_log_scalar(name, value):
    if run is not None:
        run[name].append(value)

def log_n(key, val):
    if run is not None:
        try:
            run[key].append(val)
        except Exception:
            pass

def log_pool_weights():
    if run is None: return
    wi = compute_weights_int()
    wd = compute_weights_def()
    for i, p in enumerate(DIFF_POOLS_INT):
        log_n(f'pool_int/weight_{p["name"]}', float(wi[i]))
    for i, p in enumerate(DIFF_POOLS_DEF):
        log_n(f'pool_def/weight_{p["name"]}', float(wd[i]))

# =========================
# Training Loop (pure on-policy with soft time limit + difficulty pools)
# =========================

done_count = [[], []]     # optional short-window counters (kept for continuity in your logs)
collected_eps = 0         # how many episodes collected in current rollout
updates_done = 0          # number of PPO updates already performed
CENTER_DEFENDER = False
while updates_done < NUM_UPDATES:
    # ---- Pick a difficulty pool for THIS episode based on recent success rates ----
    w_int = compute_weights_int(step=updates_done, max_step=NUM_UPDATES)
    w_def = compute_weights_def(step=updates_done, max_step=NUM_UPDATES)
    i_pid = int(np.random.choice(len(DIFF_POOLS_INT), p=w_int))
    d_pid = int(np.random.choice(len(DIFF_POOLS_DEF), p=w_def))
    sampled_params = apply_dual_pool_params(env, i_pid, d_pid)
    log_pool_weights()
    log_n('pool_int/episode_pid', i_pid)
    log_n('pool_def/episode_pid', d_pid)
    for k, v in sampled_params.items():
        log_n(f'pool_params/{k}', float(v))

    # reset env, get obs
    if updates_done >= 5000 and updates_done % 5000 == 0:
        CENTER_DEFENDER = True
    obs_n = env.reset(CENTER_DEFENDER)
    o1, o2 = obs_n[0], obs_n[1]

    ep_r = [0.0, 0.0]
    steps_this_ep = 0
    hx_dir1, hx_dir2 = None, None
    # Rollout one episode
    while True:
        # direction labels for joint-Q critics (using env geometry for now)
        dir_def_to_intr, dir_int_to_hvt = compute_directions_from_env(env)
        # --- predictor current (always compute; may be overridden if visible)
        obs_i = torch.tensor(o1, dtype=torch.float32, device=DEVICE)
        obs_d = torch.tensor(o2, dtype=torch.float32, device=DEVICE)
        # intruder obs carries defender-relative slice at [6:10] (len 4)
        pred_def_to_intr_now, _, hx_dir1 = dir_pred_intruder.step(torch.cat((obs_i[0:4], obs_i[6:10]), dim=-1), hx_dir1)
        # defender obs carries intruder→HVT slice at [8:12] (len 4)
        pred_int_to_hvt_now, _, hx_dir2 = dir_pred_defender.step(torch.cat((obs_d[0:4], obs_d[8:12]), dim=-1), hx_dir2)

        v1_now = seen_defender_by_intruder(o1)
        v2_now = seen_intruder_by_defender(o2)

        dir_def_to_intr = dir_def_to_intr if v1_now else pred_def_to_intr_now
        dir_int_to_hvt = dir_int_to_hvt if v2_now else pred_int_to_hvt_now

        #testing provide with true action
        a1 = intruder.sample_action(o1)
        a2 = defender.sample_action(o2)
        _, logp1, val1 = select_action(intruder, o1, a2)
        _, logp2, val2 = select_action(defender, o2, a1)


        # one-hot actions for mpe
        onehot_a1 = np.eye(action_dim)[a1]
        onehot_a2 = np.eye(action_dim)[a2]

        next_obs, decomposed_r, dones, _ = env.step(np.array([onehot_a1, onehot_a2]))
        true_done = bool(np.any(dones))

        # aggregate rewards (keep your original decomposition semantics)
        r1 = float(sum(decomposed_r[0][:3]))
        r2 = float(sum(decomposed_r[1]))
        final_r_i = float(decomposed_r[0][2])
        final_r_d = float(decomposed_r[1][2])
        # Soft time limit: if not naturally done and step cap reached, force terminal with penalty
        steps_this_ep += 1
        hit_time_limit = False
        if (not true_done) and USE_SOFT_TIME_LIMIT and (steps_this_ep >= MAX_STEPS_PER_EPISODE):
            hit_time_limit = True
            true_done = True
            r1 += TIME_LIMIT_PENALTY_INT
            r2 += TIME_LIMIT_PENALTY_DEF
            # fabricate dones only for curriculum bookkeeping; training uses `true_done`
            dones = [False, False]

        insert_data(intruder, o1, a1, a2, r1, logp1, val1, true_done)
        insert_data(defender, o2, a2, a1, r2, logp2, val2, true_done)

        o1, o2 = next_obs[0], next_obs[1]
        ep_r[0] += r1
        ep_r[1] += r2

        '''access_angle = decomposed_r[0][-1]
        if access_angle is not None:
            maybe_log_scalar('train/Access_Angle', access_angle)'''

        if true_done:
            # ---- Write back success stats for this pool ----
            intr_success = 1 if final_r_i > 0.0 else 0
            def_success = 1 if final_r_d > 0.0 else 0

            POOL_STATS_INT[i_pid].episodes.append(1)
            POOL_STATS_INT[i_pid].successes.append(intr_success)
            POOL_STATS_INT[i_pid].lengths.append(steps_this_ep)

            POOL_STATS_DEF[d_pid].episodes.append(1)
            POOL_STATS_DEF[d_pid].successes.append(def_success)
            POOL_STATS_DEF[d_pid].lengths.append(steps_this_ep)

            # Optional short-window agent-level counters (kept for your existing dashboards)
            done_count[0].append(1 if dones[0] else 0)
            done_count[1].append(1 if dones[1] else 0)
            if len(done_count[0]) > 100: done_count[0].pop(0)
            if len(done_count[1]) > 100: done_count[1].pop(0)

            maybe_log_scalar('train/time_limit_forced', int(hit_time_limit))
            break

    # Episode logging
    maybe_log_scalar('train/ep_length', steps_this_ep)
    maybe_log_scalar('train/Intruder_Reward', ep_r[0])
    maybe_log_scalar('train/Defender_Reward', ep_r[1])

    # These two reference last step's decomposed_r content, as in your original code
    maybe_log_scalar('train/intruder_final_r', decomposed_r[0][2])
    maybe_log_scalar('train/defender_final_r', decomposed_r[1][2])

    maybe_log_scalar('train/adv_respawn_pos', env.world.adv_respawn_pos)
    maybe_log_scalar('train/def_respawn_pos', env.world.def_respawn_pos)

    collected_eps += 1

    # When enough episodes are collected, do ONE on-policy update for each agent, then clear buffers
    if collected_eps >= ROLLOUT_EPISODES:
        intruder_out = intruder.update(0.0)  # terminal at end of episodes → bootstrap 0
        defender_out = defender.update(0.0)

        maybe_log_scalar('rollout/intruder_actor_loss', intruder_out['actor_loss'])
        maybe_log_scalar('rollout/intruder_critic_loss', intruder_out['critic_loss'])
        maybe_log_scalar('rollout/defender_actor_loss', defender_out['actor_loss'])
        maybe_log_scalar('rollout/defender_critic_loss', defender_out['critic_loss'])

        # Clear buffers explicitly to start the next collection window
        intruder.buffer.states.clear();  intruder.buffer.actions.clear(); intruder.buffer.dirs.clear()
        intruder.buffer.logprobs.clear();intruder.buffer.values.clear()
        intruder.buffer.rewards.clear(); intruder.buffer.dones.clear()

        defender.buffer.states.clear();  defender.buffer.actions.clear(); defender.buffer.dirs.clear()
        defender.buffer.logprobs.clear();defender.buffer.values.clear()
        defender.buffer.rewards.clear(); defender.buffer.dones.clear()

        collected_eps = 0
        updates_done += 1

        # Periodic checkpoint
        if updates_done % 50 == 0:
            torch.save(intruder.net.state_dict(), os.path.join(model_path, f"test_PPO_intruder_{exp_id}"))
            torch.save(defender.net.state_dict(), os.path.join(model_path, f"test_PPO_defender_{exp_id}"))

# Final save
torch.save(intruder.net.state_dict(), os.path.join(model_path, f"test_PPO_intruder_{exp_id}"))
torch.save(defender.net.state_dict(), os.path.join(model_path, f"test_PPO_defender_{exp_id}"))
