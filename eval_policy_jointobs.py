'''====================================================================================
Load learned policies, play HVT and record episodes for (later) visualization.
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
NUM_EPISODES = 5
STEPS_PER_EPISODE = 200

scenario='eot/simple_hvt_1v1_random_orig'
envs=make_env(scenario_name=scenario, logging=True, done=True)
n_features = 24 #Joint observations for now
n_actor_actions = 5
device = 'cpu' if not CUDA else 'cuda'

actor1 = ActorNetwork("act1", n_features, n_actor_actions, 1.0, 1.0, cuda=CUDA)
actor2 = ActorNetwork("act2", n_features, n_actor_actions, 1.0, 1.0, cuda=CUDA)
#==============Load defender's best policy, and attacker's contemporary policy======================
actor1.net.load_state_dict(torch.load("jt_act1"))
actor2.net.load_state_dict(torch.load("best_act2")) #BiB: Also select def/att @ bottom

for ep in range(NUM_EPISODES):
    o1, o2 = envs.reset()
    print("HVT @",envs.world.landmarks[0].state.p_pos)
    ep_r = 0
    o1t = torch.cat([torch.tensor(o1, dtype=torch.float, device=device), \
                     torch.tensor(o2, dtype=torch.float, device=device)], dim=-1)
    o2, o1 = torch.cat([torch.tensor(o1, dtype=torch.float, device=device), \
                        torch.tensor(o2, dtype=torch.float, device=device)], dim=-1), o1t
    a1 = actor1.sample_action( o1 )
    a2 = actor2.sample_action( o2 )

    #p_obs1, p_obs2 = noisy_private_obs(a1.detach(), a2.detach())
    #prior1, prior2 = bf1.prior, bf2.prior
    #_, prior1, pa2 = bf1.update(p_obs1, prior1) # Outputs predicted action of other agent
    #_, prior2, pa1 = bf2.update(p_obs2, prior2)
    for step in range(STEPS_PER_EPISODE):
        (o1_, o2_), r, done, info = envs.step(np.array([np.eye(n_actor_actions)[a1], np.eye(n_actor_actions)[a2]])) # Step in environment
        # =================This records last 5 episodes=====================================================
        if NUM_EPISODES-ep<=5:
            envs.log(ep, step+1,(o1_, o2_), r, done, info) #Need unmodified o1, o2
        #===================================================================================================
        o1_t = torch.cat([torch.tensor(o1_, dtype=torch.float, device=device), \
                          torch.tensor(o2_, dtype=torch.float, device=device)], dim=-1)
        o2_, o1_ = torch.cat([torch.tensor(o1_, dtype=torch.float, device=device), \
                              torch.tensor(o2_, dtype=torch.float, device=device)], dim=-1), o1_t
        a1_ = actor1.sample_action( o1_)
        a2_ = actor2.sample_action( o2_)
        if np.any(done):
            print(f'Done occurred in episode {ep}, step {step}: {done}. Rewards={r}')
            break
        else:
            (o1, o2), (a1, a2) = (o1_, o2_), (a1_, a2_)

import os
current_path=os.getcwd()
logdir="log"
full_path=os.path.join(current_path,logdir)
for k in  envs.logger.logs.keys():
    filename = f"logfile_{k}_def" #BiB: change def/att based on whose best policy was loaded
    print(full_path+"/"+filename)
    if os.path.exists(full_path+"/"+filename+".csv"):
        os.remove(full_path+"/"+filename+".csv")
    envs.logger.save(k,path=full_path,filename=filename)
