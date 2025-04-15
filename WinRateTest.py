import sys, time
import numpy as np
from ac_nets import *
from belief_filter import BeliefFilter
from multiagent_particle_env.make_env import make_env
from multiagent_particle_env.scenarios.eot.simple_hvt_1v1_random import Scenario
from multiagent_particle_env.logger import Logger

CUDA=True
NUM_EPISODES = 5000
STEPS_PER_EPISODE = 100

scenario='eot/simple_hvt_1v1_random_orig'
envs=make_env(scenario_name=scenario, logging=True, done=True)
n_features = 12 #Joint observations for now
n_actor_actions = 5
device = 'cpu' if not CUDA else 'cuda'

actor1 = ActorNetwork("act1", n_features, n_actor_actions, 1.0, 1.0, cuda=CUDA)
actor2 = ActorNetwork("act2", n_features, n_actor_actions, 1.0, 1.0, cuda=CUDA)
#==============Load defender's best policy, and attacker's contemporary policy======================
actor1.net.load_state_dict(torch.load("TrainResult/0.85_RandAtt_Def_0.2Pt_0.1Pa_0.3Hvt#281/last_act1"))
actor2.net.load_state_dict(torch.load("TrainResult/0.85_RandAtt_Def_0.2Pt_0.1Pa_0.3Hvt#281/last_act2")) #BiB: Also select def/att @ bottom
DefWinCount = 0
AttWinCount = 0
EvenCount = 0
for ep in range(NUM_EPISODES):
    o1, o2 = envs.reset()
    print("HVT @",envs.world.landmarks[0].state.p_pos)
    ep_r = 0
    o1 = torch.tensor(o1, dtype=torch.float, device=device)
    o2 = torch.tensor(o2, dtype=torch.float, device=device)
    a1 = actor1.sample_action( o1 )
    a2 = actor2.sample_action( o2 )

    #p_obs1, p_obs2 = noisy_private_obs(a1.detach(), a2.detach())
    #prior1, prior2 = bf1.prior, bf2.prior
    #_, prior1, pa2 = bf1.update(p_obs1, prior1) # Outputs predicted action of other agent
    #_, prior2, pa1 = bf2.update(p_obs2, prior2)
    ep_r = [0, 0]
    for step in range(STEPS_PER_EPISODE):
        (o1_, o2_), r, done, info = envs.step(np.array([np.eye(n_actor_actions)[a1], np.eye(n_actor_actions)[a2]])) # Step in environment
        # =================This records last 5 episodes=====================================================
        if NUM_EPISODES-ep<=5:
            envs.log(ep, step+1,(o1_, o2_), r, done, info) #Need unmodified o1, o2
        #===================================================================================================
        o1_ = torch.tensor(o1_, dtype=torch.float, device=device)

        o2_ = torch.tensor(o2_, dtype=torch.float, device=device)
        a1_ = actor1.sample_action( o1_)
        a2_ = actor2.sample_action( o2_)
        ep_r[0] += sum(r[0])
        ep_r[1] += sum(r[1])
        if np.any(done):
            print(f'Done occurred in episode {ep}, step {step}: {done}. Rewards={ep_r}')
            if done[0] == True:
                AttWinCount += 1
            else:
                DefWinCount += 1
            break
        else:
            (o1, o2), (a1, a2) = (o1_, o2_), (a1_, a2_)
    if not np.any(done):
        EvenCount += 1

att_win_rate = AttWinCount/NUM_EPISODES
def_win_rate = DefWinCount/NUM_EPISODES
even_rate = EvenCount/NUM_EPISODES
print('Intruder win rate is: ', att_win_rate, ' in ', NUM_EPISODES, ' runs.' )
print('Defender win rate is: ', def_win_rate, ' in ', NUM_EPISODES, ' runs.' )
print('Even rate is: ', even_rate, ' in ', NUM_EPISODES, ' runs.' )