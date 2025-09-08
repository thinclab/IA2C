# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
simple_hvt_1v1_random.py

Updated and Enhanced version of OpenAI Multi-Agent Particle Environment
(https://github.com/openai/multiagent-particle-envs)
"""

from array import array
import numpy as np
import math
import pandas as pd
from multiagent_particle_env.core import World, Agent, Landmark
from multiagent_particle_env.scenario import BaseScenario

__author__ = 'Rolando Fernandez'
__copyright__ = 'Copyright 2020, Multi-Agent Particle Environment'
__credits__ = ['Rolando Fernandez', 'OpenAI']
__license__ = ''
__version__ = '0.0.1'
__maintainer__ = 'Rolando Fernandez'
__email__ = 'rolando.fernandez1.civ@army.mil'
__status__ = 'Dev'
import random

#declare global variable
ATT_Sensing_region = 0.25
HVT_Size = 0.15
HVT_Sensing_region = 0.6
V = 0.7
Speed_Factor = 0.5
Map_size = 1
Agent_size = 0.05
class Scenario(BaseScenario):
    """
    Define the world, reward, and observations for the scenario.
    """

    def __init__(self):
        # Debug verbose output
        self.debug = False

    def make_world(self): #, args): BiB: args not used
        """
        Construct the world

        Returns:
            world (multiagent_particle_env.core.World): World object with agents and landmarks
        """
        # Debug verbose output
        self.debug = False

        # Boolean variable to detect if intruder enter HVT sensing range
        self.in_HVT = False
        # Create world and set properties
        world = World()
        world.dimension_communication = 2
        world.log_headers = ["Agent_Type", "Fixed", "Perturbed", "X", "Y", "dX", "dY", "fX", "fY", "Collision"]
        world.map_size = 1
        world.provide_global_obs = False
        # Defender and Attacker
        num_agents = 2

        # High Value Target (HVT)
        num_landmarks = 1

        # Add agents
        world.agents = [Agent() for i in range(num_agents)]

        # All agents and the HVT have the same base size
        # size is in mm?
        factor = 0.25
        size = 0.025 * factor
        '''
        # Standard
        attacker_size = size
        attacker_sense_region_size = size * (20 / factor)

        defender_size = size
        defender_sense_region_size = size * (4 / factor)

        hvt_size = size + defender_sense_region_size
        hvt_sense_region_size = size * (20 / factor)

        # 50% of the HVT size
        defender_size = hvt_size * 0.5
        '''
        #---------------------BiB------------------------
        attacker_size = size #1/160
        attacker_sense_region_size = ATT_Sensing_region#size * (4 / factor) #1/10

        defender_size = size
        defender_sense_region_size = size * (4 / factor) #1/10

        hvt_size = HVT_Size#size + defender_sense_region_size #17/160
        hvt_sensing_region = HVT_Sensing_region
        hvt_sense_region_size = hvt_size + hvt_sensing_region#size * (20 / factor) #


        # 50% of the HVT size
        #defender_size = hvt_size * 0.5
        #------------------------------------------------

        # Attacker
        world.agents[0].name = 'agent {}'.format(1)
        world.agents[0].adversary = True
        world.agents[0].accel = 3.0
        world.agents[0].collide = True
        world.agents[0].color = np.array([0.85, 0.35, 0.35])
        world.agents[0].has_sense = True
        world.agents[0].silent = True
        world.agents[0].sense_region = attacker_sense_region_size
        world.agents[0].size = Agent_size
        world.agents[0].max_speed = Speed_Factor * V
        world.adv_respawn_pos = HVT_Size + HVT_Sensing_region / 2 + 0.15

        # Defender
        # Sense region for defender is 20% smaller because
        # it has a speed advantage over the attacker
        world.agents[1].name = 'agent {}'.format(0)
        world.agents[1].accel = 4.0
        world.agents[1].collide = True
        world.agents[1].color = np.array([0.35, 0.85, 0.35])
        world.agents[1].has_sense = True
        world.agents[1].sense_region = 0.0
        world.agents[1].silent = True
        world.agents[1].sense_region = defender_sense_region_size
        world.agents[1].size = Agent_size
        world.agents[1].max_speed = Speed_Factor

        # Add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]

        # High Value Target (HVT)
        # Sense region for HVT is 20% larger because it cannot move
        world.landmarks[0].name = 'landmark {}'.format(0)
        world.landmarks[0].boundary = False
        world.landmarks[0].collide = False
        world.landmarks[0].color = np.array([0.25, 0.25, 0.25])
        world.landmarks[0].has_sense = True
        world.landmarks[0].movable = False
        world.landmarks[0].sense_region = hvt_sense_region_size
        world.landmarks[0].size = hvt_size

        # Add boundary landmarks
        world.landmarks = world.landmarks + world.set_dense_boundaries()

        # Make initial conditions
        self.reset_world(world, 0)

        return world
    def engage_phase_pos(self):

        angle = random.uniform(0, 2 * math.pi)

        adv_pos_y = 1
        adv_pos_x = 1
        intruder_sensing_range = 0.1
        HVT_sensing_range = 0.5
        adv_pos_x = HVT_sensing_range * math.cos(angle)
        adv_pos_y = HVT_sensing_range * math.sin(angle)
        adv_pos = [adv_pos_x, adv_pos_y]
        angle = random.uniform(0, 2 * math.pi)
        def_pos_x = 1
        def_pos_y = 1
        while abs(def_pos_y) > HVT_sensing_range or abs(def_pos_x) > HVT_sensing_range:
            angle = random.uniform(0, 2 * math.pi)
            def_pos_x = adv_pos_x + intruder_sensing_range * math.cos(angle)
            def_pos_y = adv_pos_y + intruder_sensing_range * math.sin(angle)
        def_pos = [def_pos_x, def_pos_y]
        return adv_pos, def_pos

    def asym_phase_pos(self, Intruder_pos=None, Defender_pos=True):
        HVT_size = HVT_Size
        sensing_range = HVT_Sensing_region
        HVT_sensing_range = HVT_size + sensing_range
        intruder_sensing_region = ATT_Sensing_region
        intruder_x, intruder_y = 0, 0
        defender_x, defender_y = 0, 0
        if Intruder_pos == None:
            sample_dist = random.uniform(HVT_sensing_range**2, (HVT_sensing_range + intruder_sensing_region)**2)
            rho_intruder = math.sqrt(sample_dist)
            theta_intruder = random.uniform(0, 2*math.pi)
            intruder_x = rho_intruder * math.cos(theta_intruder)
            intruder_y = rho_intruder * math.sin(theta_intruder)
        else:
            intruder_x = Intruder_pos[0]
            intruder_y = Intruder_pos[1]

        while Defender_pos == True:
            #rho_defender = intruder_sensing_region * math.sqrt(random.random())
            theta_defender = random.uniform(0, 2*math.pi)
            defender_x = intruder_x + intruder_sensing_region * math.cos(theta_defender)
            defender_y = intruder_y + intruder_sensing_region * math.sin(theta_defender)

            if defender_x ** 2 + defender_y ** 2 <= HVT_sensing_range ** 2:
                break

        return [intruder_x, intruder_y], [defender_x, defender_y]

    def sample_in_quarter_disk(self, radius=0.75):
        radius = HVT_Size+HVT_Sensing_region + 0.2
        u = random.random()
        r = radius
        theta = random.uniform(0, math.pi / 2)
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        return [x, y]

    def random_agents_inside_HVT(self, inside_HVT):
        '''
        0: defender randomly generate inside HVT
        1: defender randomly generate outside HVT but inside HVT sensing range
        2: defender randomly generate inside HVT sening range
        '''
        HVT_size = HVT_Size
        sensing_range = HVT_Sensing_region
        HVT_sensing_range = HVT_size + sensing_range
        low_bound = -HVT_size #HVT size
        high_bound = HVT_size
        rng = random.Random()
        #rng.seed(42)
        pos_x = 100
        if inside_HVT == 0:
            pos_y = rng.uniform(low_bound, high_bound)
            while pos_x * pos_x + pos_y * pos_y > pow(high_bound, 2):
                pos_x = rng.uniform(low_bound, high_bound)  # * (1 if random.randint(0, 1) == 0 else -1)
            return [pos_x, pos_y]
        elif inside_HVT == 1:
            pos_y = rng.uniform(-1*HVT_sensing_range, HVT_sensing_range)
            while pos_x * pos_x + pos_y * pos_y > HVT_sensing_range * HVT_sensing_range or pos_x * pos_x + pos_y * pos_y < pow(high_bound, 2):
                pos_x = rng.uniform(-1*HVT_sensing_range, HVT_sensing_range)
            return [pos_x, pos_y]
        elif inside_HVT == 2:
            pos_y = rng.uniform(-1*HVT_sensing_range, HVT_sensing_range)
            while pos_x * pos_x + pos_y * pos_y > HVT_sensing_range * HVT_sensing_range:
                pos_x = rng.uniform(-1*HVT_sensing_range, HVT_sensing_range)
            return [pos_x, pos_y]

    def random_agents_with_constrain_intruer(self, world, restart_in_HVT):
        HVT_size = HVT_Size
        sensing_range = HVT_Sensing_region
        HVT_sensing_range = HVT_size + sensing_range + 0.1
        rng = random.Random()
        if restart_in_HVT and world.adv_respawn_pos < HVT_sensing_range:
            world.adv_respawn_pos += 0.00
        def inside_HVT():
            theta = np.random.uniform(0, 2*np.pi)

            #r = np.sqrt(np.random.uniform(self.adv_respawn_pos**2, (HVT_size + sensing_range)**2))
            r = world.adv_respawn_pos
            pos_x = r * np.cos(theta)
            pos_y = r * np.sin(theta)
            return [pos_x, pos_y]

        def outside_HVT():
            #intruder pos initial
            low_bound = -Map_size#map max size
            high_bound = Map_size
            pos_x, pos_y = 0,0
            dist_sq = (pos_x)**2 + (pos_y)**2
            while dist_sq < HVT_sensing_range**2:
                pos_x = np.random.uniform(low_bound, high_bound)
                pos_y = np.random.uniform(low_bound, high_bound)
                dist_sq = (pos_x)**2 + (pos_y)**2

            return [pos_x, pos_y]
        if world.adv_respawn_pos < HVT_sensing_range:
            return inside_HVT()
        else:
            return outside_HVT()

    def reset_world(self, world, restart_in_HVT):
        """
        Reset the world to the initial conditions.

        Args:
            world (multiagent_particle_env.core.World): World object with agents and landmarks
        """
        # Set random initial states for HVT
        for landmark in world.landmarks:
            if not landmark.boundary:
                #landmark.state.p_pos = np.random.uniform(-0.39, +0.39, world.dimension_position)
                landmark.state.p_pos = np.zeros(world.dimension_position) #BiB: changed to fix @ origin
                landmark.state.p_vel = np.zeros(world.dimension_position)

        # TODO: Set sudo random postions for defender and attacker dependent on HVT

        # Set random initial states for agents
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dimension_position)
            agent.state.p_vel = np.zeros(world.dimension_position)
            agent.state.c = np.zeros(world.dimension_communication)

        #BiB: set fixed initial locations
        #pos = self.engage_phase_pos()
        #random.seed()
        self.in_HVT = False
        #asym_pos_att, asym_pos_def = self.asym_phase_pos(Intruder_pos=[0.55, 0.5], Defender_pos=True)
        world.agents[0].state.p_pos = np.asarray(self.random_agents_with_constrain_intruer(world, restart_in_HVT))#np.asarray(pos[0])#np.asarray(self.random_agents_with_constrain_intruer())#np.asarray([0.6, 0.7])#np.asarray(self.random_agents_with_constrain_intruer())
        world.agents[1].state.p_pos = np.asarray([0.0, 0.0])#np.asarray(pos[1])#np.asarray(self.random_agents_inside_HVT(2))#np.asarray([0.0, 0.0])#np.asarray(self.random_agents_inside_HVT(False))

    def good_agents(self, world):
        """
        Returns all agents that are not adversaries in a list.

        Returns:
            (list) All the agents in the world that are not adversaries.
        """
        return [agent for agent in world.agents if not agent.adversary]

    def angle_asym(self, center, intruder, defender):
        xc, yc = center
        x1, y1 = intruder
        x2, y2 = defender

        # vectors from center
        v1 = (x1 - xc, y1 - yc)
        v2 = (x2 - xc, y2 - yc)

        # dot product and norms
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        norm = math.hypot(*v1) * math.hypot(*v2)

        # guard against divide-by-zero or tiny rounding errors
        cos_theta = max(-1.0, min(1.0, dot / norm))
        theta = math.acos(cos_theta)
        return theta

    def angle_xaxis(self, intruder):
        pos = intruder.state.p_pos
        angle = math.atan2(pos[1], pos[0])
        return angle * 180 / math.pi

    def enter_HVT(self, world):
        hvt = [landmark for landmark in world.landmarks if not landmark.boundary][0]
        intruder = self.adversaries(world)[0]
        defender = self.good_agents(world)[0]
        angle = None
        if not self.in_HVT:
            if world.in_sense_region(hvt, intruder):
                #angle = self.angle_asym([0, 0], intruder.state.p_pos, defender.state.p_pos)
                angle = self.angle_xaxis(intruder)
                self.in_HVT = True
        else:
            if not world.in_sense_region(hvt, intruder):
                self.in_HVT = False
        return angle

    def adversaries(self, world):
        """
        Returns all agents that are adversaries in a list.

        Returns:
            (list) All the agents in the world that are adversaries.
        """
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world, dense=True):
        """
        Reward is based on prey agent not being caught by predator agents.

        Good agents are negatively rewarded if caught by adversaries and for exiting the screen.

        Adversaries are rewarded for collisions with good agents.



        Dense reward at start (get other agent in sense region)

        and

        Sparse reward after


        Give small reward for getting other agent in your sense region
        Give larger reward for completion of objective




        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks
            dense (boolean): Specifies whether to use dense reward

        Returns:
            If agent is adversary:
                self.adversary_reward() result
            Else:
                self.agent_reward() result
        """
        if agent.adversary:
            return self.adversary_reward(agent, world, dense)
        else:
            return self.agent_reward(agent, world, dense)

    def agent_reward(self, agent, world, dense):
        """
        Defender reward

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks
            dense (boolean): Specifies whether to use dense reward

        Returns:
            (float) Total agent reward
        """
        reward = 0

        def align_reward_from_vel(
                v_a,
                v_b,
                a_in_mask=None,
                lam_align=0.05,
                positive_only=True,
                v_ref_b=0.05,
                min_speed_a=1e-6,
                min_speed_b=1e-6,
                eps=1e-8
        ):
            v_a = np.asarray(v_a, dtype=np.float64)
            v_b = np.asarray(v_b, dtype=np.float64)
            assert v_a.shape[-1] == 2 and v_b.shape[-1] == 2, "velocities must have last dim = 2"

            # normal v
            n_a = np.linalg.norm(v_a, axis=-1)  # [...]
            n_b = np.linalg.norm(v_b, axis=-1)  # [...]

            # cosθ ∈ [-1, 1]
            dot = (v_a * v_b).sum(axis=-1)  # [...]
            cos = dot / (n_a * n_b + eps)
            if positive_only:
                cos = np.clip(cos, 0.0, None)  # only reward if their direction is same

            # encourage defender to move
            w = 0.5
            if v_ref_b is not None:
                w_b = np.clip(n_b / (v_ref_b + eps), 0.0, 1.0)
                w = w * w_b

            r = lam_align * cos * w

            # eliminate reward when speed is too small
            valid = (n_a >= min_speed_a) & (n_b >= min_speed_b)
            r = np.where(valid, r, 0.0)
            if a_in_mask is not None:
                a_in_mask = np.asarray(a_in_mask, dtype=bool)
                r = np.where(a_in_mask, r, 0.0)

            return r

        intruder = self.adversaries(world)[0]
        landmarks = [landmark for landmark in world.landmarks if not landmark.boundary]

        # Reward can optionally be dense
        if dense:
            # Incentivize defender to remain near HVT and keep attacker away from HVT
            step_r, shape_r, final_r = 0, 0, 0
            for hvt in landmarks:
                if world.in_sense_region(hvt, agent):
                    step_r = 0.00
                    '''else:
                    reward -= 0.1'''

                if world.in_sense_region(hvt, intruder):
                    HVT_radius = hvt.sense_region + hvt.size
                    relative_pos = agent.state.p_pos - intruder.state.p_pos
                    cur_dist = pow(relative_pos[0] * relative_pos[0] + relative_pos[1] * relative_pos[1], 1/2)
                    #if cur_dist >= HVT_radius:
                    #shape_r = 0.05 * (HVT_radius - cur_dist) / HVT_radius
                    #shape_r = max(shape_r, 0)
                    #shape_r += align_reward_from_vel(intruder.state.p_vel, agent.state.p_vel, v_ref_b=0.05)
                    #HVT_intruder_dist = pow(intruder.state.p_pos[0]**2 + intruder.state.p_pos[1]**2, 1/2)
                    #shape_r -= 4*HVT_intruder_dist/HVT_radius
                    #else:
                    #    shape_r = (HVT_radius - cur_dist) / (HVT_radius) * 2
                    #shape_r = -3 * cur_dist#max(shape_r, 0)
                    #shape_r = np.exp(-2 * cur_dist) * 0.05
                        #shape_r += HVT_radius/(cur_dist + HVT_radius) * 2
                    '''else:0.3
                        reward -= 0.2

                    relative_pos = hvt.state.p_pos - intruder.state.p_pos
                    cur_dist = pow(relative_pos[0] * relative_pos[0] + relative_pos[1] * relative_pos[1], 1/2)
                    if cur_dist < 0.5:
                        reward -= (1 - cur_dist)/0.5 * 0.5'''
                '''else:
                    #leads def back to HVT
                    relative_pos = agent.state.p_pos - hvt.state.p_pos
                    cur_dist = pow(relative_pos[0] * relative_pos[0] + relative_pos[1] * relative_pos[1], 1/2)
                    reward += (0.5 - cur_dist)/0.5 * 0.5'''

        # Determine collisions with attackers, assign reward

        if world.is_collision(agent, intruder):
            final_r = 20

        # Determine Attacker collision with HVT and assign penalty

        for hvt in landmarks:
            if world.is_collision(intruder, hvt):
                final_r = -20#- ((1 - cur_dist)/1 * 2)

        # Determine if agent left the screen and assign penalties
        for coordinate_position in range(world.dimension_position):
            shape_r -= world.bound(abs(agent.state.p_pos[coordinate_position]))


        return [step_r, shape_r, final_r]

    def adversary_reward(self, agent, world, dense):
        """
        Attacker reward

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks
            dense (boolean): Specifies whether to use dense reward

        Returns:
            (float) Total agent reward
        """
        defender = self.good_agents(world)[0]
        reward = 0
        agents = self.good_agents(world)
        landmarks = [landmark for landmark in world.landmarks if not landmark.boundary]
        step_r, shape_r, final_r = 0, 0, 0

        # Reward can optionally be dense
        if dense:
            # Incentivize attacker to search out HVT
            for hvt in landmarks:
                #reward for intruder stay outside HVT sensing range
                if world.in_sense_region(hvt, agent):
                    step_r += 0.0#-0.1
                else:
                    step_r -= 0.0
                if world.in_sense_region(agent, hvt):
                    hvt_pos = hvt.state.p_pos - agent.state.p_pos
                    dist = np.sqrt(np.sum(np.square(hvt_pos))) - hvt.size - agent.size#pow(hvt_pos[0] * hvt_pos[0] + hvt_pos[1] * hvt_pos[1], 1/2)
                    #shape_r += (agent.sense_region - dist)/(agent.sense_region) * 0.1
                else:
                    step_r -= 0.05
                """if world.in_sense_region(agent, defender):
                    relative_pos = agent.state.p_pos - defender.state.p_pos
                    dist = np.sqrt(np.sum(np.square(relative_pos))) #- agent.size - defender.size
                    shape_r -= (agent.sense_region - dist) / agent.sense_region * 0.2"""
                #else:
                #    step_r -= 0



        # Determine collisions with defenders, assign penalties

        if world.is_collision(agent, defender):
            final_r = -10

        # Determine Attacker collision with HVT and assign reward
        for hvt in landmarks:
            if world.is_collision(agent, hvt):
                final_r = 25

        # Determine if agent left the screen and assign penalties
        for coordinate_position in range(world.dimension_position):
            shape_r -= world.bound(abs(agent.state.p_pos[coordinate_position]))

        enter_angle = self.enter_HVT(world)

        return [step_r, shape_r, final_r, enter_angle]

    def observation(self, agent, world):
        """
        Define the observations.

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks

        Returns:
            (np.array) Observations array with the velocity of the agent, the position of the agent,
                       distance to all landmarks in the agent's reference frame,
                       distance to all other agents in the agent's reference frame,
                       and the velocities of the good agents.
        """
        intruder = self.adversaries(world)[0]
        defender = self.good_agents(world)[0]
        hvt = None
        for landmark in world.landmarks:
            if not landmark.boundary:
                hvt = landmark
                break

        # Get positions of HVT in this agent's reference frame
        landmarks_pos = [array('d', [0, 0])]
        hvt_sense_pos = [array('d', [0, 0])]
        other_pos = [array('d', [0, 0])]
        other_vel = [array('d', [0, 0])]
        self_pos = [array('d', [0, 0])]
        self_vel = [array('d', [0, 0])]
        # obs for intruder
        #print(landmark.state.p_pos - intruder.state.p_pos)
        if agent.adversary:
            if world.in_sense_region(intruder, hvt):
                landmarks_pos = [landmark.state.p_pos - intruder.state.p_pos]
            if world.in_sense_region(intruder, defender):
                other_pos = [defender.state.p_pos - intruder.state.p_pos]
                other_vel = [defender.state.p_vel]
            self_pos = [intruder.state.p_pos]
            self_vel = [intruder.state.p_vel]

        else:
            if world.in_sense_region(hvt, intruder):
                hvt_sense_pos = [intruder.state.p_pos - hvt.state.p_pos]
                landmarks_pos = [hvt.state.p_pos - defender.state.p_pos]
                other_pos = [intruder.state.p_pos - defender.state.p_pos]
                other_vel = [intruder.state.p_vel]
            self_pos = [defender.state.p_pos]
            self_vel = [defender.state.p_vel]
        #print(self_vel , self_pos , landmarks_pos , other_pos , other_vel , hvt_sense_pos)
        return np.concatenate(self_vel + self_pos + landmarks_pos + other_pos + other_vel + hvt_sense_pos)

    '''def observation(self, agent, world):
        adversaries = self.adversaries(world)

        # Get positions of HVT in this agent's reference frame
        landmarks_pos = []
        hvt_sense_pos = []
        hvt_sense_vel = []
        other_pos = []
        other_vel = []
        for landmark in world.landmarks:
            if not landmark.boundary:
                # Defender always has position of HVT
                if not agent.adversary:
                    landmarks_pos.append(landmark.state.p_pos - agent.state.p_pos)

                    # Defender has access to HVT sense region information
                    for adv in adversaries:
                        if world.in_sense_region(landmark, adv):
                            hvt_sense_pos.append(landmark.state.p_pos - adv.state.p_pos)
                            hvt_sense_vel.append(adv.state.p_vel)
                            if agent.adversary == False:
                                other_pos.append(agent.state.p_pos - adv.state.p_pos)
                                other_vel.append(adv.state.p_vel)
                # Attacker only gets position of HVT if it is sensed
                else:
                    if world.in_sense_region(agent, landmark):
                        landmarks_pos.append(landmark.state.p_pos - agent.state.p_pos)

        # Positions, and velocities of all other agents in this agent's reference frame

        for other in world.agents:
            if other is agent:
                continue
            if agent.adversary == True:
                if world.in_sense_region(agent, other):
                    other_pos.append(other.state.p_pos - agent.state.p_pos)
                    other_vel.append(other.state.p_vel)

        if self.debug:
            print("### AGENT {} ###".format(agent.name))
            # len = 2
            print("agent.state.p_vel: {}".format(agent.state.p_vel))
            # len = 2
            print("agent.state.p_pos: {}".format(agent.state.p_pos))
            # len = 1 or 0
            print("landmarks_pos: {}".format(landmarks_pos))
            # len = 1 or 0
            print("other_pos: {}".format(other_pos))
            # len = 1 or 0
            print("other_vel: {}".format(other_vel))

        if agent.adversary:
            # Sensed both HVT and defender
            if len(other_pos) != 0 and len(landmarks_pos) != 0:
                if self.debug:
                    print("Scenario Observation: Attacker sensed both HVT and defender")
                return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] +
                                      landmarks_pos + other_pos + other_vel + [array('d', [0, 0])])
            # Sensed only the defender
            elif len(other_pos) != 0 and len(landmarks_pos) == 0:
                if self.debug:
                    print("Scenario Observation: Attacker sensed only the defender")
                return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] +
                                      [array('d', [0, 0])] + other_pos + other_vel + [array('d', [0, 0])])
            # Sensed only the HVT
            elif len(other_pos) == 0 and len(landmarks_pos) != 0:
                if self.debug:
                    print("Scenario Observation: Attacker sensed only the HVT")
                return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] +
                                      landmarks_pos + [array('d', [0, 0])] + [array('d', [0, 0])] +
                                      [array('d', [0, 0])])
            # Sensed nothing
            else:
                if self.debug:
                    print("Scenario Observation: Attacker sensed nothing")
                return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] +
                                      [array('d', [0, 0])] + [array('d', [0, 0])] +
                                      [array('d', [0, 0])] + [array('d', [0, 0])])
        else:
            # Sensed attacker with both it's and HVT's region
            if len(other_pos) != 0 and len(hvt_sense_pos) != 0:
                if self.debug:
                    print("Scenario Observation: Defender sensed the Attacker with both "
                          "it's and the HVT's sense region")
                return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] +
                                      landmarks_pos + other_pos + other_vel + hvt_sense_pos)
            # Sensed attacker with only it's region
            elif len(other_pos) != 0 and len(hvt_sense_pos) == 0:
                if self.debug:
                    print("Scenario Observation: Defender sensed the Attacker with it's sense region")
                return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] +
                                      landmarks_pos + other_pos + other_vel + [array('d', [0, 0])])
            # Sensed attacker with only HVT's region
            elif len(other_pos) == 0 and len(hvt_sense_pos) != 0:
                if self.debug:
                    print("Scenario Observation: Defender sensed the Attacker with HVT's sense region")
                return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] +
                                      landmarks_pos + other_pos + other_vel + hvt_sense_pos)
            # Sensed nothing
            else:
                if self.debug:
                    print("Scenario Observation: Defender sensed nothing")
                return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] +
                                      landmarks_pos + [array('d', [0, 0])] + [array('d', [0, 0])] +
                                      [array('d', [0, 0])])'''

    def done(self, agent, world):
        """
        Determines whether the terminal condition for the episode has been reached.

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks

        Returns:
            (bool) Terminal condition reached flag
        """
        adversaries = self.adversaries(world)
        landmarks = [landmark for landmark in world.landmarks if not landmark.boundary]
        attacker_flag = False
        defender_flag = False

        # Determine collisions with defenders, assign penalties
        if agent.adversary:
            for hvt in landmarks:
                if world.is_collision(agent, hvt):
                    attacker_flag = True
        else:
            for adv in adversaries:
                if world.is_collision(agent, adv):
                    defender_flag = True

        return attacker_flag or defender_flag

    def logging(self, agent, world):
        """
        Collect data for logging.

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks

        Returns:
            (list) Data for logging
        """
        # Log elements
        agent_type = ""
        fixed = agent.is_fixed_policy
        perturbed = agent.is_perturbed_policy
        x = agent.state.p_pos[0]
        y = agent.state.p_pos[1]
        dx = agent.state.p_vel[0]
        dy = agent.state.p_vel[1]
        fx = agent.action.u[0]
        fy = agent.action.u[1]
        collision = 0

        # Check for collisions
        good_agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if agent in good_agents:
            agent_type = "Defender"
            for adv in adversaries:
                if world.is_collision(agent, adv):
                    collision += 1
        elif agent in adversaries:
            agent_type = "Attacker"
            for ga in good_agents:
                if world.is_collision(agent, ga):
                    collision += 1
        else:
            collision = "N/A"

        log_data = [agent_type, fixed, perturbed, x, y, dx, dy, fx, fy, collision]

        return log_data
