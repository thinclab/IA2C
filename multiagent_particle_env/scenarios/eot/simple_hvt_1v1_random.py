# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
simple_hvt_1v1_random.py

Updated and Enhanced version of OpenAI Multi-Agent Particle Environment
(https://github.com/openai/multiagent-particle-envs)
"""

from array import array
import numpy as np

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


class Scenario(BaseScenario):
    """
    Define the world, reward, and observations for the scenario.
    """

    def __init__(self):
        # Debug verbose output
        self.debug = False

    def make_world(self): #, args):
        """
        Construct the world

        Returns:
            world (multiagent_particle_env.core.World): World object with agents and landmarks
        """
        # Debug verbose output
        self.debug = False

        # Create world and set properties
        world = World()
        world.dimension_communication = 2
        world.log_headers = ["Agent_Type", "Fixed", "Perturbed", "X", "Y", "dX", "dY", "fX", "fY", "Collision"]

        # Defender and Attacker
        num_agents = 2

        # High Value Target (HVT)
        num_landmarks = 1

        # Add agents
        world.agents = [Agent() for i in range(num_agents)]

        # All agents and the HVT have the same base size
        # size is in mm?
        #    # All agents and the HVT have the same base size
        # size is in mm?
        factor = 0.25
        size = 0.025 * factor

        # Standard
        attacker_size = size
        attacker_sense_region_size = size * (20 / factor)

        defender_size = size
        defender_sense_region_size = size * (4 / factor)

        hvt_size = size + defender_sense_region_size
        hvt_sense_region_size = size * (20 / factor)

        # 50% of the HVT size
        defender_size = hvt_size * 0.5

        # Attacker
        world.agents[0].name = 'agent {}'.format(1)
        world.agents[0].adversary = True
        world.agents[0].accel = 3.0
        world.agents[0].collide = True
        world.agents[0].color = np.array([0.85, 0.35, 0.35])
        world.agents[0].has_sense = True
        world.agents[0].silent = True
        world.agents[0].sense_region = attacker_sense_region_size
        world.agents[0].size = attacker_size
        world.agents[0].max_speed = 1.0

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
        world.agents[1].size = defender_size
        world.agents[1].max_speed = 1.3

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

    def reset_world(self, world):
        """
        Reset the world to the initial conditions.

        Args:
            world (multiagent_particle_env.core.World): World object with agents and landmarks
        """
        # Set random initial states for HVT
        for landmark in world.landmarks:
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.39, +0.39, world.dimension_position)
                landmark.state.p_vel = np.zeros(world.dimension_position)

        # TODO: Set sudo random postions for defender and attacker dependent on HVT

        # Set random initial states for agents
        # for agent in world.agents:
        #     agent.state.p_pos = np.random.uniform(-1, +1, world.dimension_position)
        #     agent.state.p_vel = np.zeros(world.dimension_position)
        #     agent.state.c = np.zeros(world.dimension_communication)
        
        
        
        # this section writing by me,replace the upward section, to limit the defender on the circle of TSR    # 
        world.agents[0].state.p_pos = np.array([0., 0.6]) #np.random.uniform(-1, +1, world.dimension_position)
        #print("---here", world.agents[0].state.p_pos)
        world.agents[0].state.p_vel = np.zeros(world.dimension_position)
        world.agents[0].state.c = np.zeros(world.dimension_communication)

        x1,x2=np.random.uniform(-0.5, +0.5,2)
        # print("the cirlce of hvt sense size is 0.5, initial point should be in the cirle")
        while(x1**2+x2**2>0.5**2):
            x1,x2=np.random.uniform(-0.5, +0.5,2) # out of loop until the x1,x2 in the cirlce with radius of 0.5
        world.agents[1].state.p_pos = [0,0] #[x1,x2]
        world.agents[1].state.p_vel = np.zeros(world.dimension_position)
        world.agents[1].state.c = np.zeros(world.dimension_communication)
        
        
        
        
        

    def good_agents(self, world):
        """
        Returns all agents that are not adversaries in a list.

        Returns:
            (list) All the agents in the world that are not adversaries.
        """
        # print("good_agents sets:",[agent for agent in world.agents if not agent.adversary])
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
        # print("bad_agents sets:",self.adversary_reward(agent, world, dense))
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
        adversaries = self.adversaries(world)
        landmarks = [landmark for landmark in world.landmarks if not landmark.boundary]
        
        # Reward can optionally be dense
        if dense:
            # Incentivize defender to remain near HVT and keep attacker away from HVT
            for hvt in landmarks:
                if world.in_sense_region(hvt, agent):
                    reward += 0.1

        # Determine collisions with attackers, assign reward
        for adv in adversaries:
            if agent.collide:
                if world.is_collision(agent, adv):
                    reward += 10

        # Determine Attacker collision with HVT and assign penalty
        for adv in adversaries:
            for hvt in landmarks:
                if world.is_collision(adv, hvt):
                    reward -= 10

        # Determine if agent left the screen and assign penalties
        for coordinate_position in range(world.dimension_position):
            reward -= world.bound(abs(agent.state.p_pos[coordinate_position]))

        return reward

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
        reward = 0
        agents = self.good_agents(world)
        landmarks = [landmark for landmark in world.landmarks if not landmark.boundary]
        
        # Reward can optionally be dense
        if dense:
            # Incentivize attacker to search out HVT
            for hvt in landmarks:
                if not world.in_sense_region(hvt, agent):
                    reward -= 0.1

        # Determine collisions with defenders, assign penalties
        for ag in agents:
            if agent.collide:
                if world.is_collision(agent, ag):
                    reward -= 10
                    # print("reward is 10,collision.")

        # Determine Attacker collision with HVT and assign reward
        for hvt in landmarks:
            if world.is_collision(agent, hvt):
                reward += 10
                # print("hvt:",hvt)print(s1.__dir__())
                # print("hvt property:===========",hvt.__dir__(),"reward is 10:=================")
                # print("reward is 10,collision")
                # should be stop the trajactory imedately

        # Determine if agent left the screen and assign penalties
        for coordinate_position in range(world.dimension_position):
            reward -= world.bound(abs(agent.state.p_pos[coordinate_position]))
            # print("reward:",reward)
            # print("world bound abs:",abs(agent.state.p_pos[coordinate_position]))

        return reward

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

        # Get positions of HVT in this agent's reference frame
        landmarks_pos = []
        hvt_sense_pos = []
        hvt_sense_vel = []
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

                # Attacker only gets position of HVT if it is sensed
                else:
                    if world.in_sense_region(agent, landmark):
                        landmarks_pos.append(landmark.state.p_pos - agent.state.p_pos)

        # Positions, and velocities of all other agents in this agent's reference frame
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent:
                continue
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
                                      landmarks_pos + [array('d', [0, 0])] + [array('d', [0, 0])] + hvt_sense_pos)
            # Sensed nothing
            else:
                if self.debug:
                    print("Scenario Observation: Defender sensed nothing")
                return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] +
                                      landmarks_pos + [array('d', [0, 0])] + [array('d', [0, 0])] +
                                      [array('d', [0, 0])])

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
