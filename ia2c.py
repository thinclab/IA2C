'''====================================================================================
Implementation of IA2C using Actor-Critic Network classes in Org domain
where episodes have a fixed max length.

Copyright (C) May, 2024  Bikramjit Banerjee

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

===================================================================================='''

import sys
import numpy as np
import gymnasium as gym
from ac_nets import *
from belief_filter import BeliefFilter

CUDA=False
LR_C =0.0002
LR_A =0.0001
BETA = 0.001
GAMMA = 0.9
NUM_EPISODES = 5000
STEPS_PER_EPISODE = 30

from gymnasium.envs.registration import register
register(
     id="Org-v0",
     entry_point="Org:Org",
     max_episode_steps=30,
)

n_envs=10
if __name__ == '__main__':    
    envs = gym.make_vec("Org-v0", num_envs=n_envs)
    n_features = 6 #env.observation_space.shape[0]
    n_actor_actions = 3 #env.action_space.n
    n_critic_actions = 9
    n_models = 5
    critic1 = CriticNetwork("crit1", n_features, n_critic_actions, LR_C, cuda=CUDA)
    critic2 = CriticNetwork("crit2", n_features, n_critic_actions, LR_C, cuda=CUDA)
    actor1 = ActorNetwork("act1", n_features, n_actor_actions, LR_A, BETA, cuda=CUDA)
    actor2 = ActorNetwork("act2", n_features, n_actor_actions, LR_A, BETA, cuda=CUDA)
    device = 'cpu' if not CUDA else 'cuda'

    def noisy_private_obs(a1, a2):
        p_obs1, p_obs2 = np.ones((n_envs, n_actor_actions))*0.1, np.ones((n_envs, n_actor_actions))*0.1
        for i in range(n_envs):
            p_obs1[i,a2[i]]=0.8 #A2's action is A1's private observation
            p_obs2[i,a1[i]]=0.8
        return p_obs1, p_obs2

    reward_lst = []
    bf1, bf2 = BeliefFilter(n_models, n_actor_actions, n_envs), BeliefFilter(n_models, n_actor_actions, n_envs)
    for ep in range(NUM_EPISODES):
        obs, next_obs, reward = \
            torch.zeros(STEPS_PER_EPISODE, n_envs, n_features, device=device),\
            torch.zeros(STEPS_PER_EPISODE, n_envs, n_features, device=device),\
            torch.zeros(STEPS_PER_EPISODE, n_envs, device=device)
        true_action, predicted_action, true_next_action, predicted_next_action = \
            torch.zeros(STEPS_PER_EPISODE, n_envs, 2, device=device),\
            torch.zeros(STEPS_PER_EPISODE, n_envs, 2, device=device),\
            torch.zeros(STEPS_PER_EPISODE, n_envs, 2, device=device),\
            torch.zeros(STEPS_PER_EPISODE, n_envs, 2, device=device)
        s, _ = envs.reset()
        ep_r = 0

        a1 = actor1.sample_action( torch.tensor(s, device=device), grad=True )
        a2 = actor2.sample_action( torch.tensor(s, device=device), grad=True )

        p_obs1, p_obs2 = noisy_private_obs(a1.detach(), a2.detach())
        prior1, prior2 = bf1.prior, bf2.prior
        pa2, prior1, _ = bf1.update(p_obs1, prior1) # Outputs predicted action of other agent
        pa1, prior2, _ = bf2.update(p_obs2, prior2)

        for step in range(STEPS_PER_EPISODE):
            a = a1 * n_actor_actions + a2 % n_actor_actions # Joint action
            s_, r, done, term, info = envs.step(a.detach()) # Step in environment
            a1_ = actor1.sample_action( torch.tensor(s_, device=device), grad=True ) 
            a2_ = actor2.sample_action( torch.tensor(s_, device=device), grad=True ) 

            p_obs1, p_obs2 = noisy_private_obs(a1_.detach(), a2_.detach())
            pa2_, prior1, _ = bf1.update(p_obs1, prior1) #Predicted action of other agent
            pa1_, prior2, _ = bf2.update(p_obs2, prior2)

            obs[step], next_obs[step] = torch.tensor(s), torch.tensor(s_)
            true_action[step] = torch.cat([a1.unsqueeze(-1), a2.unsqueeze(-1)], dim=1)
            predicted_action[step] = torch.cat([torch.tensor(pa1).unsqueeze(-1), torch.tensor(pa2).unsqueeze(-1)], dim=1)
            true_next_action[step] = torch.cat([a1_.unsqueeze(-1), a2_.unsqueeze(-1)], dim=1)
            predicted_next_action[step] = torch.cat([torch.tensor(pa1_).unsqueeze(-1), torch.tensor(pa2_).unsqueeze(-1)], dim=1) 

            reward[step] =  torch.tensor(r)

            s, a1, a2, pa1, pa2 = s_, a1_, a2_, pa1_, pa2_
            ep_r += r

        nja1 = true_next_action[:,:,0].int() * n_actor_actions + predicted_next_action[:,:,1].int() % n_actor_actions
        nja2 = predicted_next_action[:,:,0].int() * n_actor_actions + true_next_action[:,:,1].int() % n_actor_actions
        ep_next_action1 = F.one_hot(nja1.long(), num_classes=n_critic_actions).float()
        ep_next_action2 = F.one_hot(nja2.long(), num_classes=n_critic_actions).float()
        Q_next1 = (critic1.run_main( next_obs, grad=True ) * ep_next_action1.detach()).sum(-1, keepdims=True)
        Q_next2 = (critic2.run_main( next_obs, grad=True ) * ep_next_action2.detach()).sum(-1, keepdims=True)
        critic_target1 = ( reward.unsqueeze(-1) + GAMMA * Q_next1 )
        critic_target2 = ( reward.unsqueeze(-1) + GAMMA * Q_next2 )
        jt_action = true_action[:,:,0].detach().int() * n_actor_actions + true_action[:,:,1].detach().int() % n_actor_actions
        critic1.batch_update(obs, jt_action, critic_target1) # Deviates from paper!!
        critic2.batch_update(obs, jt_action, critic_target2)

        Q_next1 = (critic1.run_main( next_obs ) * ep_next_action1).sum(-1, keepdims=True)
        Q_next2 = (critic2.run_main( next_obs ) * ep_next_action2).sum(-1, keepdims=True)
        adv_target1 = ( reward.unsqueeze(-1) + GAMMA * Q_next1 )
        adv_target2 = ( reward.unsqueeze(-1) + GAMMA * Q_next2 )
        ja1 = true_action[:,:,0].int() * n_actor_actions + predicted_action[:,:,1].int() % n_actor_actions
        ja2 = predicted_action[:,:,0].int() * n_actor_actions + true_action[:,:,1].int() % n_actor_actions
        Q1 = critic1.run_main( obs )
        Q2 = critic2.run_main( obs )
        ep_action1 = F.one_hot(ja1.long(), num_classes=n_critic_actions).float()
        ep_action2 = F.one_hot(ja2.long(), num_classes=n_critic_actions).float()
        adv1 = adv_target1 - (Q1 * ep_action1).sum(-1, keepdims=True)
        adv2 = adv_target2 - (Q2 * ep_action2).sum(-1, keepdims=True)
        actor1.batch_update( obs, true_action[:,:,0], adv1 )
        actor2.batch_update( obs, true_action[:,:,1], adv2 )

        reward_lst.append(ep_r)

        if (ep %10 == 0):
            print(ep, np.mean(reward_lst[-50:]), critic1.critic_loss, critic2.critic_loss, actor1.actor_loss, actor2.actor_loss)
