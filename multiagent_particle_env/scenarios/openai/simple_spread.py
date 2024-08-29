# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
simple_spread.py

Cooperative Navigation

Communication: No
Competitive:   No

N agents, N landmarks.

Agents are rewarded based on how far any agent is from each landmark.

Agents are penalized if they collide with other agents.

So, agents have to learn to cover all the landmarks while avoiding collisions.

Updated and Enhanced version of OpenAI Multi-Agent Particle Environment
(https://github.com/openai/multiagent-particle-envs)
"""

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

    def make_world(self, args=None):
        """
        Construct the world

        Returns:
            world (multiagent_particle_env.core.World): World object with agents and landmarks
        """
        # Create world and set properties
        world = World()
        world.collaborative = True
        world.dimension_communication = 2
        num_agents = 3
        num_landmarks = 3

        # Add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent {}'.format(i)
            agent.collide = True
            agent.silent = True
            agent.size = 0.15

        # Add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark {}'.format(i)
            landmark.collide = False
            landmark.movable = False

        # Make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        """
        Reset the world to the initial conditions.

        Args:
            world (multiagent_particle_env.core.World): World object with agents and landmarks
        """
        # Set properties for agents
        for agent in world.agents:
            agent.color = np.array([0.35, 0.35, 0.85])

        # Set properties for landmarks
        for landmark in world.landmarks:
            landmark.color = np.array([0.25, 0.25, 0.25])

        # Set random initial states for agents
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dimension_position)
            agent.state.p_vel = np.zeros(world.dimension_position)
            agent.state.c = np.zeros(world.dimension_communication)

        # Set random initial states for landmarks
        for landmark in world.landmarks:
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dimension_position)
            landmark.state.p_vel = np.zeros(world.dimension_position)

    def benchmark_data(self, agent, world):
        """
        Returns data for benchmarking purposes.

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks

        Returns:
            (tuple) The agent's reward, number of collisions, total minimum distance to landmarks,
                    and number of occupied landmarks.
        """
        reward = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0

        # Determine reward and total minimum distance to landmarks
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            reward -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1

        # Determine number of collisions between agents and adjust reward
        if agent.collide:
            for a in world.agents:
                if world.is_collision(a, agent):
                    reward -= 1
                    collisions += 1

        return tuple([reward, collisions, min_dists, occupied_landmarks])

    def reward(self, agent, world):
        """
        Reward is based on minimum agent distance to each landmark, penalized for collisions.

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks

        Returns:
            (float) Total reward, negative minimum distance to all landmarks with penalties for collisions.
        """
        # Calculate the reward, total negative minimum distance to all landmarks
        reward = 0
        for landmark in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - landmark.state.p_pos))) for a in world.agents]
            reward -= min(dists)

        # Assign penalties for collisions
        if agent.collide:
            for a in world.agents:
                if world.is_collision(a, agent):
                    reward -= 1

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
                       and the communication state of all other agents.
        """
        # Get positions and colors of all landmarks in this agent's reference frame
        landmarks_pos = []
        landmarks_color = []
        for landmark in world.landmarks:
            landmarks_pos.append(landmark.state.p_pos - agent.state.p_pos)
            landmarks_color.append(landmark.color)

        # Communication and positions of all other agents in this agent's reference frame
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + landmarks_pos + other_pos + comm)
