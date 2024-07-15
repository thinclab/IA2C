'''====================================================================================
Implementation of A2C to test Actor-Critic Network classes on RL vectorized envs
with discrete states and actions (here Taxi), and episodes have a max length.

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
import gymnasium as gym
from ac_nets import *
from gymnasium.envs.registration import register

n_envs = 10
n_updates = 2000
n_steps_per_update = 100
cuda=False

if __name__ == '__main__':
    envs = gym.make_vec("Taxi-v3", num_envs=n_envs)
    gamma = 0.99
    ent_coef = 0.01  # coefficient for the entropy bonus (to encourage exploration)
    actor_lr = 0.0001
    critic_lr = 0.0005
    obs_shape = envs.single_observation_space.n
    action_shape = envs.single_action_space.n
    critic = CriticNetwork("main1", obs_shape, action_shape, critic_lr, cuda=cuda)
    actor = ActorNetwork("act1", obs_shape, action_shape, actor_lr, ent_coef, cuda=cuda)
    device = 'cpu' if not cuda else 'cuda'
    critic_losses = []
    actor_losses = []
    entropies = []
    print(action_shape)
    for ep in range(n_updates):
        # we don't have to reset the envs, they just continue playing
        # until the episode is over and then reset automatically

        # reset lists that collect experiences of an episode
        ep_states = torch.zeros(n_steps_per_update, n_envs, obs_shape, device=device)
        ep_next_states = torch.zeros(n_steps_per_update, n_envs, obs_shape, device=device)
        ep_actions = torch.zeros(n_steps_per_update, n_envs, 1, device=device)
        ep_next_actions = torch.zeros(n_steps_per_update, n_envs, 1, device=device)
        ep_rewards = torch.zeros(n_steps_per_update, n_envs, device=device)
        masks = torch.zeros(n_steps_per_update, n_envs, device=device)

        # at the start of training reset all envs to get an initial state
        if ep == 0:
            states, info = envs.reset(seed=42)
            oh_states = F.one_hot(torch.tensor(states).long(), num_classes=obs_shape).float().to(device)
            print(oh_states)
            actions = actor.sample_action( oh_states, grad=True )
        print("ACTIONS:", actions)
        # play n steps in our parallel environments to collect data
        for step in range(n_steps_per_update):
            
            next_states, rewards, terminated, truncated, infos = envs.step(
                actions.detach().cpu().numpy()
            )
            oh_next_states = F.one_hot(torch.tensor(next_states, device=device).long(), num_classes=obs_shape).float()
            next_actions = actor.sample_action( oh_next_states, grad=True )
            ep_rewards[step] = torch.tensor(rewards, device=device)
            ep_states[step] = oh_states
            ep_next_states[step] = oh_next_states
            ep_actions[step] = actions.unsqueeze(-1)
            ep_next_actions[step] = next_actions.unsqueeze(-1)
            # add a mask (for the return calculation later);
            # for each env the mask is 1 if the episode is ongoing and 0 if it is terminated (not by truncation!)
            masks[step] = torch.tensor([not term for term in terminated])
            oh_states, actions = oh_next_states, next_actions

        # calculate the losses for actor and critic, and update them
        critic_targ = critic.run_main(ep_next_states, grad=True)
        ep_next_actions = F.one_hot(ep_next_actions.squeeze(-1).long(), num_classes=action_shape).detach().float()
        critic_target = ep_rewards.unsqueeze(-1) + gamma * masks.unsqueeze(-1) * (critic_targ * ep_next_actions).sum(-1, keepdims=True)
        critic.batch_update(ep_states, ep_actions, critic_target)
        
        Q = critic.run_main(ep_states, grad=False)
        d = actor.action_distribution(ep_states, grad=True)
        V = (Q * d).sum(-1,keepdims=True)
        ep_actions_oh = F.one_hot(ep_actions.squeeze(-1).long(), num_classes=action_shape).float()
        Q_cur = (Q * ep_actions_oh).sum(-1, keepdims=True)
        actor.batch_update(ep_states, ep_actions, (Q_cur - V))
        print(ep, ep_rewards.sum(0).mean(), critic.critic_loss, actor.actor_loss)
