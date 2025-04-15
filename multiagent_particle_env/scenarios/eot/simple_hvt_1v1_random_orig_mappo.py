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
from multiagent_particle_env.core import World, Landmark
from mappo_ac_net import MAPPOAgent as Agent
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
        CUDA = True
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
        agents = []
        num_agents = 2
        action_dim = 5
        local_obs_dim = 12
        global_obs_dim = 15  # besides local obs, add another agents relative location and speed
        total_action_dim = action_dim * action_dim  # Since there are two agents, possible joint actions could be 25
        device = 'cpu' if not CUDA else 'cuda'
        for agent_id in range(num_agents):
            agent = Agent(
                agent_id=agent_id,
                local_obs_dim=local_obs_dim,
                global_obs_dim=global_obs_dim,
                action_dim=action_dim,
                total_action_dim=total_action_dim,
                device=device)
            agents.append(agent)
        world.agents = agents#[Agent() for i in range(num_agents)]

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
        attacker_sense_region_size = 0.1#size * (4 / factor) #1/10

        defender_size = size
        defender_sense_region_size = size * (4 / factor) #1/10

        hvt_size = 0.2#size + defender_sense_region_size #17/160
        hvt_sensing_region = 0.15
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
        world.agents[0].size = 0.04#attacker_size
        world.agents[0].max_speed = 0.425

        # Defender
        # Sense region for defender is 20% smaller because
        # it has a speed advantage over the attacker
        world.agents[1].name = 'agent {}'.format(0)
        world.agents[1].accel = 4.0
        world.agents[1].collide = True
        world.agents[1].color = np.array([0.35, 0.85, 0.35])
        world.agents[1].has_sense = True
        world.agents[1].sense_region = 0.1
        world.agents[1].silent = True
        world.agents[1].sense_region = defender_sense_region_size
        world.agents[1].size = 0.04#defender_size
        world.agents[1].max_speed = 0.5

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
        self.reset_world(world)

        return world
    def engage_phase_pos(self):

        angle = random.uniform(0, 2 * math.pi)

        adv_pos_y = 1
        adv_pos_x = 1
        intruder_sensing_range = 0.3
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

    def random_agents_inside_HVT(self, inside_HVT):
        '''
        0: defender randomly generate inside HVT
        1: defender randomly generate outside HVT but inside HVT sensing range
        2: defender randomly generate inside HVT sening range
        '''
        HVT_size = 0.2
        sensing_range = 0.15
        HVT_sensing_range = HVT_size + sensing_range
        low_bound = -HVT_size #HVT size
        high_bound = HVT_size
        rng = random.Random()
        #rng.seed(42)
        pos_x = 100
        if inside_HVT == 0:
            pos_y = rng.uniform(low_bound, high_bound)
            while pos_x * pos_x + pos_y * pos_y > pow(HVT_sensing_range, 2):
                pos_x = rng.uniform(-1 * HVT_sensing_range, HVT_sensing_range)  # * (1 if random.randint(0, 1) == 0 else -1)
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

    def random_agents_with_constrain_intruer(self):
        HVT_size = 0.2
        sensing_range = 0.15
        HVT_sensing_range = HVT_size + sensing_range + 0.1
        rng = random.Random()
        #rng.seed(42)
        def inside_HVT():
            pos_y = rng.uniform(-0.5, 0.5)
            pos_x = 1
            #defender_sense_region_size = size * (4 / factor)
            HVT_SIZE = 0.025*0.25 *(4/0.25) + 0.1
            while pos_x * pos_x + pos_y * pos_y > 0.25:# or pos_x * pos_x + pos_y * pos_y < HVT_SIZE * HVT_SIZE:
                pos_x = rng.uniform(-0.5, 0.5)  # * (1 if random.randint(0, 1) == 0 else -1)
            return [pos_x, pos_y]

        def outside_HVT():
            #intruder pos initial
            low_bound = -1#map max size
            high_bound = 1
            pos_x = 0
            pos_y = np.random.uniform(low_bound, high_bound)
            if rng.randint(0, 1) == 0:
                pos_x = rng.uniform(low_bound, -1*HVT_sensing_range)
            else:
                pos_x = rng.uniform(HVT_sensing_range, high_bound)
            while (pos_x*pos_x + pos_y*pos_y) <= pow(HVT_sensing_range, 2) or (pos_x*pos_x + pos_y*pos_y) >= pow(high_bound, 2):
                pos_y = rng.uniform(low_bound, high_bound)

            return [pos_x, pos_y]
        '''if random.random() > 0.5:
            return inside_HVT()
        else:'''
        return outside_HVT()

    def reset_world(self, world):
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
        world.agents[0].state.p_pos = np.asarray(self.random_agents_with_constrain_intruer())#np.asarray(pos[0])#np.asarray(self.random_agents_with_constrain_intruer())#np.asarray([0.6, 0.7])#np.asarray(self.random_agents_with_constrain_intruer())
        world.agents[1].state.p_pos = np.asarray(self.random_agents_inside_HVT(2))#np.asarray(pos[1])#np.asarray(self.random_agents_inside_HVT(False))#np.asarray([0.0, 0.0])#np.asarray(self.random_agents_inside_HVT(False))

    def good_agents(self, world):
        """
        Returns all agents that are not adversaries in a list.

        Returns:
            (list) All the agents in the world that are not adversaries.
        """
        return [agent for agent in world.agents if not agent.adversary]

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
        intruder = self.adversaries(world)[0]
        landmarks = [landmark for landmark in world.landmarks if not landmark.boundary]

        # Reward can optionally be dense
        if dense:
            # Incentivize defender to remain near HVT and keep attacker away from HVT
            step_r, shape_r, final_r = 0, 0, 0
            for hvt in landmarks:
                if world.in_sense_region(hvt, agent):
                    step_r = 0.05
                '''else:
                    reward -= 0.1'''

                if world.in_sense_region(hvt, intruder):
                    HVT_radius = hvt.sense_region + hvt.size
                    relative_pos = agent.state.p_pos - intruder.state.p_pos
                    cur_dist = pow(relative_pos[0] * relative_pos[0] + relative_pos[1] * relative_pos[1], 1/2)
                    shape_r = (HVT_radius * 2 - cur_dist) / (HVT_radius * 2) * 2
                    #shape_r += HVT_radius/(cur_dist + HVT_radius) * 2
                    '''else:
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
            final_r = 10

        # Determine Attacker collision with HVT and assign penalty

        for hvt in landmarks:
            if world.is_collision(intruder, hvt):
                final_r = -10 #- ((1 - cur_dist)/1 * 2)

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
                if not world.in_sense_region(hvt, agent):
                    step_r += 0#-0.1
                if world.in_sense_region(agent, hvt):
                    hvt_pos = hvt.state.p_pos - agent.state.p_pos
                    dist = np.sqrt(np.sum(np.square(hvt_pos))) - hvt.size - agent.size#pow(hvt_pos[0] * hvt_pos[0] + hvt_pos[1] * hvt_pos[1], 1/2)
                    shape_r = (agent.sense_region - dist)/(agent.sense_region) * 2
                #else:
                #    step_r -= 0



        # Determine collisions with defenders, assign penalties

        if world.is_collision(agent, defender):
            final_r = -10

        # Determine Attacker collision with HVT and assign reward
        for hvt in landmarks:
            if world.is_collision(agent, hvt):
                final_r = 10

        # Determine if agent left the screen and assign penalties
        for coordinate_position in range(world.dimension_position):
            shape_r -= world.bound(abs(agent.state.p_pos[coordinate_position]))

        return [step_r, shape_r, final_r]

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
        adversaries = self.adversaries(world)
        # Get global obs when HVT sense Attacker and Attacker sense HVT and Defender
        global_obs = None
        if world.provide_global_obs == True:
            Defender = self.good_agents(world)[0]
            Attker = adversaries[0]
            for landmark in world.landmarks:
                if not landmark.boundary:
                    if world.in_sense_region(Attker, Defender) and world.in_sense_region(Attker, landmark):
                        # each agent pos
                        defender_pos = Defender.state.p_pos
                        attacker_pos = Attker.state.p_pos
                        # position between HVT and two agents
                        landmark_pos_def = landmark.state.p_pos - Defender.state.pos
                        landmark_pos_adv = landmark.state.p_pos - Attker.state.pos
                        # position between attacker and Defender
                        defender_pos_adv = Defender.state.p_pos - Attker.state.p_pos
                        attacker_pos_def = Attker.state.p_pos - Defender.state.p_pos
                        # velocity of each agent
                        attacker_vel = Attker.state.p_vel
                        defender_vel = Defender.state.p_vel
                        global_obs = np.concatenate([defender_pos] + [attacker_pos] + [landmark_pos_def] + [landmark_pos_adv] +
                                              [defender_pos_adv] + [attacker_pos_def] + [attacker_vel] + [defender_vel])





        # Get positions of HVT in this agent's reference frame
        landmarks_pos = []
        hvt_sense_pos = []
        hvt_sense_vel = []
        other_pos = []
        other_vel = []
        global_obs = []
        for landmark in world.landmarks:
            if not landmark.boundary:
                # Defender always has position of HVT
                if not agent.adversary:
                    landmarks_pos.append(landmark.state.p_pos - agent.state.p_pos)
                    #landmarks_pos.append(landmark.state.p_pos)

                    # Defender has access to HVT sense region information
                    for adv in adversaries:
                        if world.in_sense_region(landmark, adv):
                            hvt_sense_pos.append(landmark.state.p_pos - adv.state.p_pos)#change relative position order
                            hvt_sense_vel.append(adv.state.p_vel)
                            other_pos.append(adv.state.p_pos - agent.state.p_pos)
                            #other_pos.append(adv.state.p_pos)
                            other_vel.append(adv.state.p_vel)




                # Attacker only gets position of HVT if it is sensed
                else:
                    if world.in_sense_region(agent, landmark):
                        landmarks_pos.append(landmark.state.p_pos - agent.state.p_pos)
                        #landmarks_pos.append(landmark.state.p_pos)
                        #if world

        # Positions, and velocities of all other agents in this agent's reference frame

        for other in world.agents:
            if other is agent:
                continue
            if world.in_sense_region(agent, other) and agent.adversary:
                other_pos.append(other.state.p_pos - agent.state.p_pos)
                #other_pos.append(other.state.p_pos)
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
                                      landmarks_pos + other_pos + other_vel + [array('d', [0, 0])]), global_obs
            # Sensed only the defender
            elif len(other_pos) != 0 and len(landmarks_pos) == 0:
                if self.debug:
                    print("Scenario Observation: Attacker sensed only the defender")
                return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] +
                                      [array('d', [0, 0])] + other_pos + other_vel + [array('d', [0, 0])]), global_obs
            # Sensed only the HVT
            elif len(other_pos) == 0 and len(landmarks_pos) != 0:
                if self.debug:
                    print("Scenario Observation: Attacker sensed only the HVT")
                return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] +
                                      landmarks_pos + [array('d', [0, 0])] + [array('d', [0, 0])] +
                                      [array('d', [0, 0])]), global_obs
            # Sensed nothing
            else:
                if self.debug:
                    print("Scenario Observation: Attacker sensed nothing")
                return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] +
                                      [array('d', [0, 0])] + [array('d', [0, 0])] +
                                      [array('d', [0, 0])] + [array('d', [0, 0])]), global_obs
        else:
            # Sensed attacker with both it's and HVT's region
            if len(other_pos) != 0 and len(hvt_sense_pos) != 0:
                if self.debug:
                    print("Scenario Observation: Defender sensed the Attacker with both "
                          "it's and the HVT's sense region")
                return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] +
                                      landmarks_pos + other_pos + other_vel + hvt_sense_pos), global_obs
            # Sensed attacker with only it's region
            elif len(other_pos) != 0 and len(hvt_sense_pos) == 0:
                if self.debug:
                    print("Scenario Observation: Defender sensed the Attacker with it's sense region")
                return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] +
                                      landmarks_pos + other_pos + other_vel + [array('d', [0, 0])]), global_obs
            # Sensed attacker with only HVT's region
            elif len(other_pos) == 0 and len(hvt_sense_pos) != 0:
                if self.debug:
                    print("Scenario Observation: Defender sensed the Attacker with HVT's sense region")
                return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] +
                                      landmarks_pos + [array('d', [0, 0])] + [array('d', [0, 0])] + hvt_sense_pos), global_obs
            # Sensed nothing
            else:
                if self.debug:
                    print("Scenario Observation: Defender sensed nothing")
                return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] +
                                      landmarks_pos + [array('d', [0, 0])] + [array('d', [0, 0])] +
                                      [array('d', [0, 0])]), global_obs

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
