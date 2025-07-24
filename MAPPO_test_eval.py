from multiagent_particle_env.make_env import make_env
from MAPPO_net_test import PPOAgent
import torch
import numpy as np

CUDA=True
NUM_EPISODES = 5
STEPS_PER_EPISODE = 200
num_agents = 2
n_actor_actions = 5
global_obs_dim = 16
action_dim = 5
exp_id = '42'

scenario='eot/simple_hvt_1v1_random_orig_mappo'
envs=make_env(scenario_name=scenario, logging=True, done=True)
intruder = PPOAgent(global_obs_dim, action_dim)
defender = PPOAgent(global_obs_dim, action_dim)
intruder.net.load_state_dict(torch.load("test_PPO_intruder_"+exp_id))
defender.net.load_state_dict(torch.load("test_PPO_defender_"+exp_id))

for ep in range(NUM_EPISODES):
    o1, o2 = envs.reset()
    ep_r = [0, 0]
    for step in range(STEPS_PER_EPISODE):
        a1, logp1, val1 = intruder.select_action(o1)
        a2, logp2, val2 = defender.select_action(o2)
        next_obs, decomposed_r, dones, info = envs.step(np.array([np.eye(action_dim)[a1], np.eye(action_dim)[a2]]))
        done = False
        if np.any(dones):
            done = True
        r = [sum(decomposed_r[0][:3]), sum(decomposed_r[1])]
        envs.log(ep, step+1,(next_obs[0], next_obs[1]), r, dones, info) #Need unmodified o1, o2

        o1 = next_obs[0]
        o2 = next_obs[1]
        ep_r[0] += sum(decomposed_r[0][:3])
        ep_r[1] += sum(decomposed_r[1])

        if done:
            print(ep, step, ep_r)
            break
    if step >= 199:
        print(f'Not done with reward {ep, step, ep_r}')


import os
current_path=os.getcwd()
logdir="log"
full_path=os.path.join(current_path,logdir)
'''print(f"Intruder's action predictor accuracy: {predict_action2/true_actions}")
print(f"Defender's action predictor accuracy: {predict_action1/true_actions}")'''

for k in  envs.logger.logs.keys():
    filename = f"logfile_{k}_mappo_test" #BiB: change def/att based on whose best policy was loaded
    print(full_path+"/"+filename)
    if os.path.exists(full_path+"/"+filename+".csv"):
        os.remove(full_path+"/"+filename+".csv")
    envs.logger.save(k,path=full_path,filename=filename)