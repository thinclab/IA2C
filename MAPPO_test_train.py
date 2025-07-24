from multiagent_particle_env.make_env import make_env
import torch
import numpy as np
import torch.nn as nn, torch.optim as optim
from torch.distributions import Categorical
from MAPPO_net_test import PPOAgent
import matplotlib.pyplot as plt
import neptune
from API_token import api_token, mappo_project

scenario='eot/simple_hvt_1v1_random_orig_mappo'

run = False
CUDA=True
exp_id = '42'
num_agents = 2
action_dim = 5
agents = []
local_obs_dim = 12
global_obs_dim = 16 #besides local obs, add another agents relative location and speed
total_action_dim = action_dim * action_dim #Since there are two agents, possible joint actions could be 25
NUM_EPISODES = 15000
NUM_STEP = 200
n_models = 4
n_envs = 1
device = 'cpu' if not CUDA else 'cuda'
intruder_loss = []
defender_loss = []
ep_reward = [[], []]

envs=make_env(scenario_name=scenario, logging=True, done=True)
if run:
    run = neptune.init_run(project=mappo_project, api_token=api_token, tags=['no reset'])


intruder = PPOAgent(global_obs_dim, action_dim)
defender = PPOAgent(global_obs_dim, action_dim)

def insert_data(agent, obs, act, reward, logp, val, done):
    agent.buffer.states.append(obs)
    agent.buffer.actions.append(act)
    agent.buffer.logprobs.append(logp)
    agent.buffer.values.append(val)
    agent.buffer.rewards.append(reward)
    agent.buffer.dones.append(float(done))

def check_loss_convergence(loss_history, window=4, tol=1e-3):
    n = len(loss_history)
    if n < 2 * window:
        return False, None, None

    prev_mean = np.mean(loss_history[-2*window:-window])
    curr_mean = np.mean(loss_history[-window:])
    return (abs(prev_mean - curr_mean) < tol), prev_mean, curr_mean

for ep in range(NUM_EPISODES):
    o1, o2 = envs.reset()
    ep_r = [0, 0]

    for step in range(NUM_STEP):
        a1, logp1, val1 = intruder.select_action(o1)
        a2, logp2, val2 = defender.select_action(o2)
        #a2 = 0
        next_obs, decomposed_r, dones, _ = envs.step(np.array([np.eye(action_dim)[a1], np.eye(action_dim)[a2]]))
        done = False
        if np.any(dones):
            done = True
        insert_data(intruder, o1, a1, sum(decomposed_r[0][:3]), logp1, val1, done)
        insert_data(defender, o2, a2, sum(decomposed_r[1]), logp2, val2, done)

        o1 = next_obs[0]
        o2 = next_obs[1]
        ep_r[0] += sum(decomposed_r[0][:3])
        ep_r[1] += sum(decomposed_r[1])
        access_angle = decomposed_r[0][-1]
        if access_angle != None and run != False:
            run[f'train/Access_Angle'].append(access_angle)
        if done:
            o1, o2 = envs.reset()
    print(ep, step, ep_r)
    if run:
        run[f'train/Intruder_Reward'].append(ep_r[0])
        run[f'train/Defender_Reward'].append(ep_r[1])

    with torch.no_grad():
        _, next_val1 = intruder.net(torch.from_numpy(o1).float())
        _, next_val2 = defender.net(torch.from_numpy(o2).float())

    intruder_loss.append(intruder.update(next_val1.item()))
    defender_loss.append(defender.update(next_val2.item()))
    if len(intruder_loss) > 150:
        del intruder_loss[0]
        del defender_loss[0]


    if run:
        run[f'train/Intruder_Actor_Loss'].append(intruder_loss[-1]['actor_loss'])
        run[f'train/Intruder_Critic_Loss'].append(intruder_loss[-1]['critic_loss'])
        run[f'train/Intruder_Entropy'].append(intruder_loss[-1]['entropy'])
        run[f'train/Defender_Actor_Loss'].append(defender_loss[-1]['actor_loss'])
        run[f'train/Defender_Critic_Loss'].append(defender_loss[-1]['critic_loss'])
        run[f'train/Defender_Entropy'].append(defender_loss[-1]['entropy'])

    converge_intruder_actor, _, _ = check_loss_convergence([loss['actor_loss'] for loss in intruder_loss])
    converge_intruder_critic, pre_mean, cur_mean = check_loss_convergence([loss['critic_loss'] for loss in intruder_loss])
    print(f"Intruder last step mean loss {pre_mean}, cur step mean loss {cur_mean}")
    converge_defender_actor, _, _ = check_loss_convergence([loss['actor_loss'] for loss in defender_loss])
    converge_defender_critic, pre_mean, cur_mean = check_loss_convergence([loss['critic_loss'] for loss in defender_loss])
    print(f"Defender last step mean loss {pre_mean}, cur step mean loss {cur_mean}")

    if converge_defender_critic and converge_defender_actor and converge_intruder_critic and converge_intruder_actor:
        print("All agents converge")
        break
    if ep % 50 == 0:
        torch.save(intruder.net.state_dict(), "test_PPO_intruder_" + exp_id)
        torch.save(defender.net.state_dict(), "test_PPO_defender_" + exp_id)
#print(intruder_loss)
torch.save(intruder.net.state_dict(), "test_PPO_intruder_" + exp_id)
torch.save(defender.net.state_dict(), "test_PPO_defender_" + exp_id)

