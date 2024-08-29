# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
simple.py

Keep Away

Communication: No
Competitive:   No

Single agent sees landmark position, rewarded based on how close it gets to landmark.
Not a multi-agent environment -- used for debugging policies.

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
        # Create world
        world = World()

        # Add agents
        world.agents = [Agent() for i in range(1)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent {}'.format(i)
            agent.collide = False
            agent.index = i
            agent.silent = True

        # Add landmarks
        world.landmarks = [Landmark() for i in range(1)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark {}'.format(i)
            landmark.collide = False
            landmark.index = i
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
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25, 0.25, 0.25])

        # Set properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            if i == 0:
                landmark.color = np.array([0.75, 0.25, 0.25])
            else:
                landmark.color = np.array([0.75, 0.75, 0.75])

        # Set random initial states for agents
        for agent in world.agents:
            agent.state.c = np.zeros(world.dimension_communication)
            agent.state.p_pos = np.random.uniform(-1, +1, world.dimension_position)
            agent.state.p_vel = np.zeros(world.dimension_position)

        # Set random initial states for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dimension_position)
            landmark.state.p_vel = np.zeros(world.dimension_position)

    def reward(self, agent, world):
        """
        Define the reward based on how close the agent gets to landmark

        Returns:
            (float) The negative distance squared between the agent and the landmark
        """
        dist2 = np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos))

        return -dist2

    def observation(self, agent, world):
        """
        Define the observations.

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks

        Returns:
            (np.array) Observations array with agent's velocity and positions of landmarks.
        """
        # Get positions of all entities in specified agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        return np.concatenate([agent.state.p_vel] + entity_pos)
