"""
MAPPO training script — Pure On-Policy with Rollout Buffer (episodes) + Soft Time Limit.

- Collect ROLLOUT_EPISODES (e.g., 32) full episodes per cycle
- Then run ONE on-policy PPO update per agent on the concatenated rollout buffer
- Clear buffers and repeat: collect -> update -> clear
- Soft time limit prevents extremely long stalemates by forcing a terminal
  when step count reaches MAX_STEPS_PER_EPISODE (with a small terminal penalty)

Assumptions
-----------
- `PPOAgent` exposes `buffer` with fields: states, actions, logprobs, values,
  rewards, dones; and a method `update(next_value)` that computes GAE/returns.
- We set `next_value=0.0` at update time because each episode in the rollout
  ends with done=1 (either natural or soft time limit forced), i.e., no bootstrap
  across episode boundaries is required.
- Comments are in English.
"""

import os
import random
from typing import Tuple

import numpy as np
import torch

from multiagent_particle_env.make_env import make_env
from MAPPO_net_test import PPOAgent
import neptune
from API_token import api_token, mappo_project

# =========================
# Config
# =========================
scenario = 'eot/simple_hvt_1v1_random_orig_mappo'

USE_NEPTUNE = True
RUN_TAGS = ['onpolicy-rollout-buffer', 'soft-time-limit']

CUDA = True
DEVICE = 'cuda' if (CUDA and torch.cuda.is_available()) else 'cpu'

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE == 'cuda':
    torch.cuda.manual_seed_all(SEED)

# PPO / env params
num_agents = 2
action_dim = 5
obs_dim = 12

NUM_EPISODES = 15000
ROLLOUT_EPISODES = 32   # collect this many episodes before each PPO update

# Soft time-limit settings
USE_SOFT_TIME_LIMIT   = True
MAX_STEPS_PER_EPISODE = 5000     # force end if exceeded and not naturally done
TIME_LIMIT_PENALTY_INT = -0.2    # terminal shaping penalty for intruder
TIME_LIMIT_PENALTY_DEF = -0.2    # terminal shaping penalty for defender

# logging / saving
exp_id = 118
model_path = f"/home/lzeng/Thinclab Code/HVT/IA2C/mappo_training_result/{exp_id}/"
os.makedirs(model_path, exist_ok=False)

# env
env = make_env(scenario_name=scenario, logging=True, done=True)
env.world.adv_respawn_pos = 0.5

# =========================
# Agents
# =========================
intruder = PPOAgent(obs_dim, action_dim, gamma=0.98)
defender = PPOAgent(obs_dim, action_dim, gamma=0.95)

# =========================
# Utils
# =========================

def insert_data(agent: PPOAgent, obs: np.ndarray, act: int, reward: float, logp: float, val: float, done: bool):
    agent.buffer.states.append(obs)
    agent.buffer.actions.append(act)
    agent.buffer.logprobs.append(logp)
    agent.buffer.values.append(val)
    agent.buffer.rewards.append(reward)
    agent.buffer.dones.append(1.0 if done else 0.0)

@torch.no_grad()
def select_action(agent: PPOAgent, obs: np.ndarray) -> Tuple[int, float, float]:
    """Returns (action, logprob, value)."""
    a, logp, v = agent.select_action(obs)
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

# =========================
# Training Loop (pure on-policy with soft time limit)
# =========================

done_count = [[], []]   # curriculum counters
collected_eps = 0       # how many episodes collected in current rollout

for ep in range(NUM_EPISODES):
    # Curriculum (same thresholds as your original code)
    if sum(done_count[1]) > 35 and sum(done_count[0]) > 35:
        env.world.adv_respawn_pos = min(env.world.adv_respawn_pos + 0.05, 0.8)
        o1, o2 = env.reset(True)
        done_count = [[], []]
    elif sum(done_count[1]) > 55:
        min_dis = 0.5
        env.world.adv_respawn_pos = max(env.world.adv_respawn_pos - 0.05, min_dis)
        o1, o2 = env.reset(True)
        done_count = [[], []]
    elif sum(done_count[0]) > 55:
        max_dis = 0.8
        env.world.adv_respawn_pos = min(env.world.adv_respawn_pos + 0.05, max_dis)
        o1, o2 = env.reset(True)
        done_count = [[], []]
    else:
        o1, o2 = env.reset(False)

    ep_r = [0.0, 0.0]
    steps_this_ep = 0

    # Rollout one episode
    while True:
        a1, logp1, val1 = select_action(intruder, o1)
        a2, logp2, val2 = select_action(defender, o2)

        next_obs, decomposed_r, dones, _ = env.step(
            np.array([np.eye(action_dim)[a1], np.eye(action_dim)[a2]])
        )
        true_done = bool(np.any(dones))

        # aggregate rewards
        r1 = float(sum(decomposed_r[0][:3]))
        r2 = float(sum(decomposed_r[1]))

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

        insert_data(intruder, o1, a1, r1, logp1, val1, true_done)
        insert_data(defender, o2, a2, r2, logp2, val2, true_done)

        o1, o2 = next_obs[0], next_obs[1]
        ep_r[0] += r1
        ep_r[1] += r2

        access_angle = decomposed_r[0][-1]
        if access_angle is not None:
            maybe_log_scalar('train/Access_Angle', access_angle)

        if true_done:
            # curriculum bookkeeping
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
    maybe_log_scalar('train/intruder_final_r', decomposed_r[0][2])
    maybe_log_scalar('train/defender_final_r', decomposed_r[1][2])
    maybe_log_scalar('train/adv_respawn_pos', env.world.adv_respawn_pos)
    maybe_log_scalar('train/agent_size', env.world.agents[0].size)

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
        intruder.buffer.states.clear();  intruder.buffer.actions.clear()
        intruder.buffer.logprobs.clear();intruder.buffer.values.clear()
        intruder.buffer.rewards.clear(); intruder.buffer.dones.clear()

        defender.buffer.states.clear();  defender.buffer.actions.clear()
        defender.buffer.logprobs.clear();defender.buffer.values.clear()
        defender.buffer.rewards.clear(); defender.buffer.dones.clear()

        collected_eps = 0

    # Periodic checkpoint
    if ep % 50 == 0:
        torch.save(intruder.net.state_dict(), os.path.join(model_path, f"test_PPO_intruder_{exp_id}"))
        torch.save(defender.net.state_dict(), os.path.join(model_path, f"test_PPO_defender_{exp_id}"))

# Final save
torch.save(intruder.net.state_dict(), os.path.join(model_path, f"test_PPO_intruder_{exp_id}"))
torch.save(defender.net.state_dict(), os.path.join(model_path, f"test_PPO_defender_{exp_id}"))
