import numpy as np
import scipy as sp
import gymnasium as gym
from ac_nets import *
from itertools import combinations_with_replacement
from Encoder_Decoder import EncoderDecoderNetwork
from operator import add
from random import randrange
from Org import Org
import matplotlib.pyplot as plt
import random

from math import log
plt.switch_backend('agg')

import os

if os.path.exists("my_secrets.py"):
    from my_secrets import project, api_token
else:
    project, api_token = None, None

run = False

if run:
    from neptune import Run
    run = Run(
        project=project,
        api_token=api_token,
        tags=['no reset']
    )


GAMMA = 0.9
NUM_AGENTS = 20
NUM_EPISODES = 20000#50000
STEPS_PER_EPISODE = 30
NOISE = 20
DEBUG = False

critic_actions = 6
count = 4
env = Org()
for i in range(2,NUM_AGENTS):
    critic_actions += count
    count+=1
n_features = 1
actor_actions = 3

exp_buff, reward_lst, pd_lst, ep_r_lst = [], [], [], []

def generate_init_D():
    #random.seed(32)
    a_self = random.randint(0, NUM_AGENTS)
    a_balance = random.randint(0, NUM_AGENTS - a_self)
    a_group = NUM_AGENTS - a_self - a_balance
    return [a_self, a_balance, a_group]


critic, actor, encoder_decoder, D, theta = [], [], [], [], []

RT = [4,1,0]

'''a_self = random.randint(0, 20)
a_balance = random.randint(0, 20 - a_self)
a_group = 20 - a_self - a_balance

D = (a_self,a_balance,a_group)'''
#Generate possible action combinations
joints = list(combinations_with_replacement("012", NUM_AGENTS))
all_cf = []
all_state = []

for i in range(NUM_AGENTS):
    critic.append([])
    actor.append([])
    encoder_decoder.append([])
    D.append(generate_init_D())
    theta.append([torch.ones(len(joints))/len(joints)])
    exp_buff.append([])

for i in range(len(joints)):
        all_cf.append([0,0,0])

for i in range(len(joints)):
    for j in range(len(joints[i])):
        all_cf[i][int(joints[i][j])]+= 1



#return configuration index
def cf_to_index(cf):
    index = all_cf.index(cf)
    return index

#map joint action to corresponding configuration
def ja_to_cf(ja):
    cf = [0,0,0]
    for i in range(len(ja)):
        cf[ja[i]]+= 1

    return cf

#add noise to private observation
def add_noise(action, noise):
    obs = [0,0,0]

    for i in range(3):
        count = 0
        while count < action[i]:
            count += 1
            random = randrange(100)
            if random < noise/2:
                obs[(i+1)%3] += 1
            elif random >= noise/2 and random < noise:
                obs[(i+2)%3] += 1
            else:
                obs[i] += 1
    return obs

def reset_exp_buffer():
    exp_buffer = []
    for i in range(NUM_AGENTS):
        exp_buffer.append([])
    return exp_buffer
#get configuration distribution from current Dirichlet distribution
def get_dist(dist):
    count = 0
    action_dist = []
    for i in range(len(joints)):
        action_dist.append(1)
    for i in joints:
        for j in i:
            action_dist[count] = action_dist[count] * dist[int(j)]
        count += 1

    s = sum(action_dist)

    for i in range(len(action_dist)):
        action_dist[i] = action_dist[i]/s

    return action_dist

def Dirchlet_denoise(D_i, noised_obs, is_current_obs):
    d = list(map(add, list(D_i), noised_obs))
    D[i] = (d[0], d[1], d[2])
    alpha = [0, 0, 0]

    for j in range(3):
        alpha[j] = round(D[i][j] * 1.28571 - D[i][(j + 1) % 3] * 0.142857 - D[i][(j + 2) % 3] * 0.142857)
        if alpha[j] <= 0:
            alpha[j] = 0.1

    pd = np.random.dirichlet(alpha, 1)
    pd = pd[0]
    if is_current_obs:
        pd_lst.append(pd)
    pa = np.random.multinomial(NUM_AGENTS, pd, 1)
    pa = pa[0].tolist()
    return pa, alpha

for i in range(NUM_AGENTS):
    critic[i] = CriticNetwork( n_features, critic_actions)

for i in range(NUM_AGENTS):
    actor[i] = ActorNetwork( n_features, actor_actions)

for i in range(NUM_AGENTS):
    #public obs size+ private obs size+ self chosen action size + len(joints)
    encoder_decoder[i] = EncoderDecoderNetwork(1 + 3 + 1 + len(joints), 16, 1, len(joints))

for ep in range(NUM_EPISODES):
    s, _, _, _ = env.reset()
    done = False
    ep_r = 0
    s = [s]
    ja = []
    reward_buffer = list(np.zeros(NUM_AGENTS))
    for i in range(NUM_AGENTS):
        ja.append(actor[i].sample_action( s ))
        #cf = ja_to_cf(ja)
    cf = ja_to_cf(ja)

    if run:
        run[f"train/individual"].append(cf[0])
        run[f"train/balance"].append(cf[1])
        run[f"train/group"].append(cf[2])

    for step in range(STEPS_PER_EPISODE):
        s_, gr, done, info = env.step(cf) # make step in environment
        s_ = [s_]
        ja_ = []
        r = []

        for i in range(NUM_AGENTS):
            #sample next step action for each agent and add to action list
            ja_.append(actor[i].sample_action( s_ ))
        #generate next step real private obs
        cf_ = ja_to_cf(ja_)

        for i in range(NUM_AGENTS):
            #Calculate reward with action
            r.append(RT[ja[i]]+gr)
            ind_r = r[-1]
            if run:
                run[f'train/agent_{i}'].append(ind_r)
            #for graph plotting
            reward_buffer[i] += r[-1]
            all_state.append(s)
            if DEBUG:
                print ("ja = ", cf, "s = ", s_)
            #generate current step noised private obs
            noised_obs = add_noise(cf, NOISE)
            #using Drichlet distribution denoise
            '''d = list(map(add, list(D[i]), noised_obs))
            D[i] = (d[0], d[1], d[2])
            alpha = [0,0,0]

            for j in range(3):
                alpha[j] = round(D[i][j]*1.28571 - D[i][(j+1)%3]*0.142857 - D[i][(j+2)%3]*0.142857)
                if alpha[j] <= 0:
                    alpha[j] = 0.1

            pd = np.random.dirichlet(alpha, 1)
            pd = pd[0]
            pd_lst.append(pd)
            pa = np.random.multinomial(NUM_AGENTS, pd, 1)
            pa = pa[0].tolist()'''
            denoised_private_obs, alpha = Dirchlet_denoise(D[i], noised_obs, True)
            #generate next step noised private obs
            noised_obs_ = add_noise(cf_, NOISE)
            #using Drichlet distribution denoise
            denoised_private_obs_, _= Dirchlet_denoise(D[i], noised_obs_, False)
            z, next_pub_obs, next_theta = encoder_decoder[i].forward(s, denoised_private_obs, ja[i], theta[i][-1])
            z_, next_pub_obs_, next_theta_ = encoder_decoder[i].forward(s_, denoised_private_obs_, ja_[i], next_theta)
            #private obs, public obs, action config, z, z', next private obs, next public obs, true private obs, alpha
            exp_buff[i].append([denoised_private_obs, s, ja[i], z, z_ , ind_r, denoised_private_obs_, next_pub_obs, s_, cf_, alpha])

        reward_lst.append(r)
        ep_r += sum(r)
        s, ja, cf = s_, ja_, cf_

        if done:
            break

    #dist = get_dist(pd)
    #for actor, critic update
    x_vec, neuron_cf_vec, actor_adv_vec, critic_target_vec, neuron_sel_vec, Q_next = [], [], [], [], [], []
    #for decoder-encoder update
    pred_next_pub_obs, next_state, pred_act_config, true_act_config, alpha_vec = [], [], [], [], []
    for i in range(NUM_AGENTS):
        x_vec.append([])
        actor_adv_vec.append([])
        critic_target_vec.append([])
        neuron_sel_vec.append([])
        Q_next.append([])
        neuron_cf_vec.append([])
        pred_next_pub_obs.append([])
        next_state.append([])
        pred_act_config.append([])
        true_act_config.append([])
        alpha_vec.append([])

    if ep % 10 == 0:
        #Critic Update
        for i in range(NUM_AGENTS):
            #private obs, public obs, action config, z, z', next private obs, next public obs, true_next_pub_obs, true next action config, dirichlet parameter
            for (private_obs, public_obs, act_config, z, z_, reward, next_private_obs, next_public_obs, true_next_pub_obs, true_next_act_config, alpha) in exp_buff[i]:
                x_vec[i].append(public_obs)
                neuron_sel_vec[i].append(act_config)
                Q_next[i] = critic[i].run_main( next_public_obs )[act_config]
                critic_target_vec[i].append( reward + GAMMA * Q_next[i] )
                neuron_cf_vec[i].append( cf_to_index(next_private_obs) )

        for i in range(NUM_AGENTS):
            critic[i].batch_update( x_vec[i], neuron_cf_vec[i], critic_target_vec[i] )

        #Actor Update
        for i in range(NUM_AGENTS):
            #for (state, action, configuration, rewards, next_state, next_action, next_configuration) in exp_buff[i]:
            for (private_obs, public_obs, act_config, z, z_, reward, next_private_obs, next_public_obs, true_next_pub_obs, true_next_act_config, alpha) in exp_buff[i]:

                Q = critic[i].run_main( public_obs )
                Q_cur = Q[act_config]
                dist = get_dist(pd_lst[i])
                V = np.dot( Q, dist )
                actor_adv_vec[i].append( Q_cur - V )

        for i in range(NUM_AGENTS):
            actor[i].batch_update( x_vec[i], neuron_sel_vec[i], actor_adv_vec[i] )

        #encoder-decoder update
        for i in range(NUM_AGENTS):
            for (private_obs, public_obs, act_config, z, z_, reward, next_private_obs, next_public_obs, true_next_pub_obs, true_next_act_config, alpha) in exp_buff[i]:
                pred_next_pub_obs[i].append(private_obs)
                next_state[i].append(true_next_pub_obs)
                pred_act_config[i].append(next_private_obs)
                true_act_config[i].append(true_next_act_config)
                alpha_vec[i].append(alpha)
        for i in range(NUM_AGENTS):
            encoder_decoder[i].batch_update(pred_next_pub_obs[i], next_state[i], pred_act_config[i], true_act_config[i], alpha_vec[i])

        exp_buff = reset_exp_buffer()
        pd_lst=[]
    ep_r_lst.append(ep_r)
    print("Episode:" ,ep, "reward:", ep_r)

plt.plot(reward_lst)
plt.title('reward_lst')
plt.savefig('reward')
plt.close()

plt.plot(all_state)
plt.title('state')
plt.savefig('state')
np.savetxt('ep_r.csv',ep_r_lst,delimiter=',')

if run:
    run.sync()
    run.stop()

