import torch
import torch.nn as nn
import torch.optim as optim
from sympy.physics.units import action
from torch.distributions import Normal
import numpy as np

from belief_filter import BeliefFilter
from multiagent_particle_env.make_env import make_env

from mappo_ac_net import MAPPOAgent

CUDA=True
num_agents = 2
action_dim = 5
agents = []
local_obs_dim = 12
global_obs_dim = 15 #besides local obs, add another agents relative location and speed
total_action_dim = action_dim * action_dim #Since there are two agents, possible joint actions could be 25
NUM_EPISODES = 1000
NUM_STEP = 150
n_models = 4
n_envs = 1
device = 'cpu' if not CUDA else 'cuda'
'''for agent_id in range(num_agents):
    agent = MAPPOAgent(
        agent_id=agent_id,
        local_obs_dim=local_obs_dim,
        global_obs_dim=global_obs_dim,
        action_dim=action_dim,
        total_action_dim=total_action_dim,
        device=device)
    agents.append(agent)'''

scenario='eot/simple_hvt_1v1_random_orig_mappo'
envs=make_env(scenario_name=scenario, logging=True, done=True)

def noisy_private_obs(a1, a2):
    p_obs1, p_obs2 = np.ones((n_envs, action_dim))*0.1, np.ones((n_envs, action_dim))*0.1
    for i in range(n_envs):
        p_obs1[i]=a2 #A2's action is A1's private observation, no noise for now
        p_obs2[i]=a1
    return p_obs1, p_obs2



for episode in range(NUM_EPISODES):
    o1, o2 = envs.reset()
    all_dones = {agent: False for agent in envs.agents}
    local_episode_data = {agent: {'obs': [], 'global_states': [], 'actions': [],
                            'log_probs': [], 'rewards': [], 'dones': [],
                            'all_actions': []}
                    for agent in range(2)}
    global_episode_data = {agent: {'obs': [], 'global_states': [], 'actions': [],
                            'log_probs': [], 'rewards': [], 'dones': [],
                            'all_actions': []}
                       for agent in range(2)}

    step = 0
    while not all_dones or step < NUM_STEP:
        # Collect data from all agents
        global_state = []
        # predicate action
        all_actions = []
        # true action
        real_actions = []

        # Get actions from all agents
        actions = {}
        for agent in envs.agents:
            agent_id = None
            if agent.adversary == True:
                local_obs, global_obs = envs.get_agent_obs('Attacker')
                agent_id = 0
            else:
                local_obs, global_obs = envs.get_agent_obs('Defender')
                agent_id = 1

            if len(global_obs) == 0:
                obs_tensor = torch.FloatTensor(local_obs).unsqueeze(0).to(device)
            else:
                obs_tensor = torch.FloatTensor(global_obs).unsqueeze(0).to(device)
            print(obs_tensor.device)
            with torch.no_grad():
                action, log_probs, entropy = envs.agents[agent_id].actor(obs_tensor)
            actions[agent_id] = action.cpu().numpy()
            if len(global_obs) == 0:
                print(len(local_obs))
                local_episode_data[agent_id]['obs'].append(local_obs)
                local_episode_data[agent_id]['log_probs'].append(log_probs)
            else:
                global_episode_data[agent_id]['obs'].append(global_obs)
                global_episode_data[agent_id]['log_probs'].append(log_probs)

        #generate predict action for defender and attacker
        noised_obs_def, noised_obs_att = noisy_private_obs(actions[0], actions[1])
        noised_obs = [noised_obs_def, noised_obs_att]
        pred_act_def, envs.agents[0].bf.prior, _ = envs.agents[0].bf.update(noised_obs_def, envs.agents[0].bf.prior)
        pred_act_att, envs.agents[1].bf.prior, _ = envs.agents[1].bf.update(noised_obs_att, envs.agents[1].bf.prior)

        # Environment step
        next_obs, decomposed_r, dones, infos = envs.step(np.array([np.eye(action_dim)[actions[0][0]], np.eye(action_dim)[actions[1][0]]]))
        rewards = [sum(decomposed_r[0]), sum(decomposed_r[1])]
        # Store transition data
        #episode_data[agent]['global_states'].append(np.array(global_state))
        for agent in envs.agents:
            if agent.adversary == True:
                agent_id = 0
                all_actions = [actions[0], pred_act_def]
            else:
                agent_id = 1
                all_actions = [pred_act_att, actions[1]]

            if len(global_obs) == 0:
                local_episode_data[agent_id]['rewards'].append(rewards[agent_id])
                local_episode_data[agent_id]['dones'].append(dones[agent_id])
                local_episode_data[agent_id]['all_actions'].append(np.concatenate(all_actions))
                local_episode_data[agent_id]['actions'].append(actions[agent_id])
            else:
                # if enter global obs phase, provide true action to agents
                all_actions[agent_id - 1] = noised_obs[agent_id]
                global_episode_data[agent_id]['rewards'].append(rewards[agent_id])
                global_episode_data[agent_id]['dones'].append(dones[agent_id])
                global_episode_data[agent_id]['all_actions'].append(np.concatenate(all_actions))
                global_episode_data[agent_id]['actions'].append(actions[agent_id])
        all_dones = dones
        step += 1
        # Update each agent
    envs.agents[0].update(local_episode_data[0], False)

    envs.agents[1].update(local_episode_data[1], False)
    if len(global_episode_data[0]['obs']) > 0:
        envs.agents[0].update(global_episode_data[0], True)
        envs.agents[1].update(global_episode_data[1], True)
    # Print training progress
    #print(f"Episode {episode + 1}, Total Reward: {total_reward:.2f}")


