'''====================================================================================
Application of IA2C using Actor-Critic Network classes in HVT domain.
Simplifying assumptions:
(1) Agents receive joint observations;
(2) Private observations are noise-free, no belief tracking;
(3) Actions are deterministic

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

import sys, time
import numpy as np
from ac_nets import *
from belief_filter import BeliefFilter
from multiagent_particle_env.make_env import make_env
from multiagent_particle_env.scenarios.eot.simple_hvt_1v1_random import Scenario
from multiagent_particle_env.logger import Logger

CUDA=True
LR_C =0.005
LR_A =0.001
BETA = 0.001
GAMMA = 0.9
NUM_EPISODES = 200000
BATCH_SIZE = 1024
STEPS_PER_EPISODE = BATCH_SIZE

scenario='eot/simple_hvt_1v1_random_orig'
envs=make_env(scenario_name=scenario, logging=True, done=True)

n_envs=1
lowest_actor_loss, lowest_critic_loss = [np.inf, np.inf]
best_rew = [-np.inf, -np.inf]
n_features = 24 #Joint observations for now
n_actor_actions = 5 #env.action_space.n
n_critic_actions = 25 #joint actions
n_models = 4
critic1 = CriticNetwork("crit1", n_features, n_critic_actions, LR_C, cuda=CUDA)
critic2 = CriticNetwork("crit2", n_features, n_critic_actions, LR_C, cuda=CUDA)
actor1 = ActorNetwork("act1", n_features, n_actor_actions, LR_A, BETA, cuda=CUDA)
actor2 = ActorNetwork("act2", n_features, n_actor_actions, LR_A, BETA, cuda=CUDA)
device = 'cpu' if not CUDA else 'cuda'

def noisy_private_obs(a1, a2):
    p_obs1, p_obs2 = np.ones((n_envs, n_actor_actions))*0.1, np.ones((n_envs, n_actor_actions))*0.1
    for i in range(n_envs):
        p_obs1[i]=a2 #A2's action is A1's private observation, no noise for now
        p_obs2[i]=a1
    return p_obs1, p_obs2

reward_lst = []
bf1, bf2 = BeliefFilter(n_models, n_actor_actions, n_envs), BeliefFilter(n_models, n_actor_actions, n_envs)
ep = 0
avg_rew = [0.,0.]
ep_r = [0.,0.]
while ep < NUM_EPISODES:
    obs, next_obs, reward = \
        torch.zeros(STEPS_PER_EPISODE, n_envs, 2, n_features, device=device),\
        torch.zeros(STEPS_PER_EPISODE, n_envs, 2, n_features, device=device),\
        torch.zeros(STEPS_PER_EPISODE, n_envs, 2, device=device)
    predicted_action, predicted_next_action = \
        torch.zeros(STEPS_PER_EPISODE, n_envs, 2, device=device),\
        torch.zeros(STEPS_PER_EPISODE, n_envs, 2, device=device)
    true_action_1, true_action_2 = \
        torch.zeros(STEPS_PER_EPISODE, n_envs, 1, device=device),\
        torch.zeros(STEPS_PER_EPISODE, n_envs, 1, device=device)
    true_next_action_1, true_next_action_2 =\
        torch.zeros(STEPS_PER_EPISODE, n_envs, 1, device=device),\
        torch.zeros(STEPS_PER_EPISODE, n_envs, 1, device=device)
    dones = torch.zeros(STEPS_PER_EPISODE, n_envs, 1, device=device)

    o1, o2 = envs.reset()

    o1t = torch.cat([torch.tensor(o1, dtype=torch.float, device=device), \
                    torch.tensor(o2, dtype=torch.float, device=device)], dim=-1)
    o2 = torch.cat([torch.tensor(o1, dtype=torch.float, device=device), \
                    torch.tensor(o2, dtype=torch.float, device=device)], dim=-1)
    o1 = o1t #Joint observations

    a1 = actor1.sample_action(o1, grad=True )
    a2 = actor2.sample_action(o2, grad=True )

    #p_obs1, p_obs2 = noisy_private_obs(a1.detach(), a2.detach())
    #prior1, prior2 = bf1.prior, bf2.prior
    #_, prior1, pa2 = bf1.update(p_obs1, prior1) # Outputs predicted action of other agent
    #_, prior2, pa1 = bf2.update(p_obs2, prior2)

    for step in range(STEPS_PER_EPISODE):
        (o1_, o2_), r, done, info = envs.step(np.array([np.eye(n_actor_actions)[a1], np.eye(n_actor_actions)[a2]])) # Step in environment
        o1_t = torch.cat([torch.tensor(o1_, dtype=torch.float, device=device), \
                    torch.tensor(o2_, dtype=torch.float, device=device)], dim=-1)
        o2_, o1_ = torch.cat([torch.tensor(o1_, dtype=torch.float, device=device), \
                    torch.tensor(o2_, dtype=torch.float, device=device)], dim=-1), o1_t

        a1_ = actor1.sample_action( o1_, grad=True )
        a2_ = actor2.sample_action( o2_, grad=True )

        #p_obs1, p_obs2 = noisy_private_obs(a1_.detach(), a2_.detach())
        #_, prior1, pa2_ = bf1.update(p_obs1, prior1) #Predicted action of other agent
        #_, prior2, pa1_ = bf2.update(p_obs2, prior2)
        #=====================make predicted actions=true actions for now===================================
        pa2, pa2_ = a2.detach(), a2_.detach()
        pa1, pa1_ = a1.detach(), a1_.detach()
        #===================================================================================================
        obs[step] = torch.cat([o1.unsqueeze(-2), o2.unsqueeze(-2)], dim=-2)
        next_obs[step] = torch.cat([o1_.unsqueeze(-2), o2_.unsqueeze(-2)], dim=-2)
        true_action_1[step], true_action_2[step] = a1, a2
        predicted_action[step] = torch.cat([torch.tensor(pa1).unsqueeze(-1), torch.tensor(pa2).unsqueeze(-1)], dim=-1)
        true_next_action_1[step], true_next_action_2[step] = a1_, a2_
        predicted_next_action[step] = torch.cat([torch.tensor(pa1_).unsqueeze(-1), torch.tensor(pa2_).unsqueeze(-1)], dim=-1)

        reward[step] =  torch.tensor(r)
        if np.any(done):
            dones[step] = 1
            ep_r[0], ep_r[1] = ep_r[0] + r[0], ep_r[1]+r[1]
            o1, o2 = envs.reset()
            o1t = torch.cat([torch.tensor(o1, dtype=torch.float, device=device), \
                             torch.tensor(o2, dtype=torch.float, device=device)], dim=-1)
            o2, o1 = torch.cat([torch.tensor(o1, dtype=torch.float, device=device), \
                            torch.tensor(o2, dtype=torch.float, device=device)], dim=-1), o1t

            a1 = actor1.sample_action(o1, grad=True )
            a2 = actor2.sample_action(o2, grad=True )
            reward_lst.append(ep_r)
            if len(reward_lst) > 500:
                del reward_lst[0]
            ep_r = [0,0]
            ep += 1
        else:
            (o1, o2), (a1, a2), (pa1, pa2) = (o1_, o2_), (a1_, a2_), (pa1_, pa2_)
            ep_r[0], ep_r[1] = ep_r[0] + r[0], ep_r[1]+r[1]

    #=====================Agent 1 update===================================================================
    nja1 = true_next_action_1.int().squeeze(-1) * n_actor_actions + predicted_next_action[:,:,1].int() % n_actor_actions
    ep_next_action1 = F.one_hot(nja1.long(), num_classes=n_critic_actions).float()
    Q_next1 = (critic1.run_main( next_obs[:,:,0,:], grad=True ) * ep_next_action1.detach()).sum(-1, keepdims=True)
    critic_target1 = ( reward[:,:,0].unsqueeze(-1) + GAMMA * Q_next1 * (1-dones))
    jt_action = true_action_1.detach().int() * n_actor_actions + true_action_2.detach().int() % n_actor_actions
    critic1.batch_update(obs[:,:,0,:], jt_action, critic_target1) # Deviates from paper!!

    Q_next1 = (critic1.run_main( next_obs[:,:,0,:] ) * ep_next_action1).sum(-1, keepdims=True)
    adv_target1 = ( reward[:,:,0].unsqueeze(-1) + GAMMA * Q_next1 * (1-dones))
    ja1 = true_action_1.int().squeeze(-1) * n_actor_actions + predicted_action[:,:,1].int() % n_actor_actions
    Q1 = critic1.run_main( obs[:,:,0,:] )
    ep_action1 = F.one_hot(ja1.long(), num_classes=n_critic_actions).float()
    adv1 = adv_target1 - (Q1 * ep_action1).sum(-1, keepdims=True)
    actor1.batch_update( obs[:,:,0,:], true_action_1, adv1 )

    
    #=====================Agent 2 update===================================================================
    nja2 = predicted_next_action[:,:,0].int() * n_actor_actions + true_next_action_2.int().squeeze(-1) % n_actor_actions
    ep_next_action2 = F.one_hot(nja2.long(), num_classes=n_critic_actions).float()
    Q_next2 = (critic2.run_main( next_obs[:,:,1,:], grad=True ) * ep_next_action2.detach()).sum(-1, keepdims=True)
    critic_target2 = ( reward[:,:,1].unsqueeze(-1) + GAMMA * Q_next2 * (1-dones))
    jt_action = true_action_1.detach().int() * n_actor_actions + true_action_2.detach().int() % n_actor_actions
    critic2.batch_update(obs[:,:,1,:], jt_action, critic_target2) # Deviates from paper!!

    Q_next2 = (critic2.run_main( next_obs[:,:,1,:] ) * ep_next_action2).sum(-1, keepdims=True)
    adv_target2 = ( reward[:,:,1].unsqueeze(-1) + GAMMA * Q_next2 * (1-dones))
    ja2 = predicted_action[:,:,0].int() * n_actor_actions + true_action_2.int().squeeze(-1) % n_actor_actions
    Q2 = critic2.run_main( obs[:,:,1,:] )
    ep_action2 = F.one_hot(ja2.long(), num_classes=n_critic_actions).float()
    adv2 = adv_target2 - (Q2 * ep_action2).sum(-1, keepdims=True)
    actor2.batch_update( obs[:,:,1,:], true_action_2, adv2 )


    reward_lst.append(ep_r)
    avg_rew = np.mean(reward_lst[-100:],axis=0)
    if avg_rew[0] > best_rew[0] and ep>100:
        print(f'Avg. episode rew of ATTACKER at new high: {avg_rew[0]}. Saving models (episode {ep})')
        torch.save(actor1.net.state_dict(), "best_act1")
        torch.save(actor2.net.state_dict(), "jt_act2")
        best_rew[0] = avg_rew[0]
    
    if avg_rew[1] > best_rew[1] and ep>100:
        print(f'Avg. episode rew of DEFENDER at new high: {avg_rew[1]}. Saving models (episode {ep})')
        torch.save(actor2.net.state_dict(), "best_act2")
        torch.save(actor1.net.state_dict(), "jt_act1")
        best_rew[1] = avg_rew[1]

    if (critic1.critic_loss + critic2.critic_loss < lowest_critic_loss) and (abs(actor1.actor_loss) + abs(actor2.actor_loss) < lowest_actor_loss):
        lowest_critic_loss = critic1.critic_loss + critic2.critic_loss
        lowest_actor_loss = abs(actor1.actor_loss) + abs(actor2.actor_loss)
        print(f'Joint losses at new low: {lowest_critic_loss},{lowest_actor_loss}. Saving both actors (episode {ep})')
        torch.save(actor1.net.state_dict(), "jt_best_act1")
        torch.save(actor2.net.state_dict(), "jt_best_act2")
    
    if (ep %100 == 99):
        print(ep, avg_rew, critic1.critic_loss, critic2.critic_loss,  actor1.actor_loss, actor2.actor_loss)
print('Best stats (best_rew, lowest_actor_loss, lowest_critic_loss):',best_rew, lowest_actor_loss, lowest_critic_loss)
