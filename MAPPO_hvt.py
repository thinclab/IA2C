import torch
import torch.nn as nn
import torch.optim as optim
from sympy.physics.units import action
from torch.distributions import Normal
import numpy as np
from API_token import api_token, mappo_project
from belief_filter import BeliefFilter
from multiagent_particle_env.make_env import make_env
import neptune
from mappo_ac_net import MAPPOAgent

run = True
CUDA=True
num_agents = 2
action_dim = 5
agents = []
local_obs_dim = 12
global_obs_dim = 15 #besides local obs, add another agents relative location and speed
total_action_dim = action_dim * action_dim #Since there are two agents, possible joint actions could be 25
NUM_EPISODES = 8000
NUM_STEP = 200
n_models = 4
n_envs = 1
device = 'cpu' if not CUDA else 'cuda'



scenario='eot/simple_hvt_1v1_random_orig_mappo'
envs=make_env(scenario_name=scenario, logging=True, done=True)
if run:
    run = neptune.init_run(project=mappo_project, api_token=api_token, tags=['no reset'])

def noisy_private_obs(a1, a2):
    p_obs1, p_obs2 = np.ones((n_envs, action_dim))*0.1, np.ones((n_envs, action_dim))*0.1
    for i in range(n_envs):
        p_obs1[i]=a2 #A2's action is A1's private observation, no noise for now
        p_obs2[i]=a1
    return p_obs1, p_obs2



for episode in range(NUM_EPISODES):
    o1, o2 = envs.reset()
    cur_obs = [o1, o2]
    all_dones = {agent: False for agent in envs.agents}
    episode_data = {agent: {'obs': [], 'global_states': [], 'actions': [],
                            'log_probs': [], 'rewards': [], 'dones': [],
                            'all_actions': []}
                    for agent in range(2)}
    
    step = 0
    ep_r = [0, 0]
    while not all_dones or step < NUM_STEP:
        # Collect data from all agents
        global_state = []
        # predicate action
        all_actions = []
        # true action
        real_actions = []
        ep_r = [0, 0]
        # Get actions from all agents
        actions = {}
        for agent_id in range(num_agents):
            obs_tensor = torch.FloatTensor(cur_obs[agent_id]).unsqueeze(0).to(device)
            #print(obs_tensor.shape)
            with torch.no_grad():
                action, log_probs, entropy = envs.agents[agent_id].actor(obs_tensor)
            actions[agent_id] = action.cpu().numpy()
            episode_data[agent_id]['obs'].append(cur_obs[agent_id])
            episode_data[agent_id]['log_probs'].append(log_probs)


        #generate predict action for defender and attacker
        noised_obs_def, noised_obs_att = noisy_private_obs(actions[0], actions[1])
        noised_obs = [noised_obs_def, noised_obs_att]
        pred_act_def, envs.agents[0].bf.prior, _ = envs.agents[0].bf.update(noised_obs_def, envs.agents[0].bf.prior)
        pred_act_att, envs.agents[1].bf.prior, _ = envs.agents[1].bf.update(noised_obs_att, envs.agents[1].bf.prior)

        # Environment step
        next_obs, decomposed_r, dones, infos = envs.step(np.array([np.eye(action_dim)[actions[0][0]], np.eye(action_dim)[actions[1][0]]]))
        ep_r[0] += sum(decomposed_r[0])
        ep_r[1] += sum(decomposed_r[1])
        if run:
            run[f'train/intruder_step_r'].append(decomposed_r[0][0])
            run[f'train/intruder_shape_r'].append(decomposed_r[0][1])
            run[f'train/intruder_final_r'].append(decomposed_r[0][2])
            run[f'train/defender_step_r'].append(decomposed_r[1][0])
            run[f'train/defender_shape_r'].append(decomposed_r[1][1])
            run[f'train/defender_final_r'].append(decomposed_r[1][2])
        rewards = [sum(decomposed_r[0]), sum(decomposed_r[1])]
        # Store transition data
        #episode_data[agent]['global_states'].append(np.array(global_state))
        for agent_id in range(num_agents):
            if envs.agents[agent_id].adversary == True:
                all_actions = [actions[0], pred_act_def]
            else:
                all_actions = [pred_act_att, actions[1]]

            episode_data[agent_id]['rewards'].append(rewards[agent_id])
            episode_data[agent_id]['dones'].append(dones[agent_id])
            episode_data[agent_id]['all_actions'].append(np.concatenate(all_actions))
            episode_data[agent_id]['actions'].append(actions[agent_id])

        all_dones = dones
        cur_obs = next_obs
        step += 1

    # Update each agent
    envs.agents[0].update(episode_data[0])
    envs.agents[1].update(episode_data[1])
    if run:
        intruder_loss, defender_loss = envs.get_loss()
        run[f'train/intruder Actor loss'].append(intruder_loss[0])
        run[f'train/intruder Critic loss'].append(intruder_loss[2])
        run[f'train/defender Actor loss'].append(defender_loss[0])
        run[f'train/defender Critic loss'].append(defender_loss[2])
        run[f'train/defender_ep_r'].append(ep_r[1])
        run[f'train/defender_ep_r'].append(ep_r[0])

    # Print training progress
    if episode % 100 == 0 or episode >= NUM_EPISODES - 1 :
        intruder_loss, defender_loss = envs.get_loss()
        print(f"Episode {episode}, Intruder Actor loss: {intruder_loss[0]}, "
              #f"Intruder Local Critic loss: {intruder_loss[1]}, "
              f"Intruder Critic loss: {intruder_loss[2]}, "
              f"Defender Actor loss: {defender_loss[0]}, "
              #f"Defender Local Critic Loss: {defender_loss[1]}, "
              f"Defender Critic Loss: {defender_loss[2]}")

        save_path = '/home/lzeng/Thinclab Code/HVT/IA2C/mappo_training_result/test'
        envs.save_model(save_path, 'test_train7')
    else:
        print(f"Episode {episode}")



