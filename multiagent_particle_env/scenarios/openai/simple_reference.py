# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
simple_reference.py

Communication: Yes
Competitive:   No

2 agents, 3 landmarks of different colors.

Each agent wants to get to their target landmark, which is known only by other agent.

Reward is collective. So agents have to learn to communicate the goal of the other agent,
and navigate to their landmark.

This is the same as the simple_speaker_listener scenario where both agents are simultaneous speakers and listeners.

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
        world.dimension_communication = 10

        # Add agents
        world.agents = [Agent() for i in range(2)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent {}'.format(i)
            agent.collide = False

        # Add landmarks
        world.landmarks = [Landmark() for i in range(3)]
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
        # Want other agent to go to the goal landmark
        world.agents[0].goal_a = world.agents[1]
        world.agents[0].goal_b = np.random.choice(world.landmarks)
        world.agents[1].goal_a = world.agents[0]
        world.agents[1].goal_b = np.random.choice(world.landmarks)

        # Set properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25, 0.25, 0.25])

        # Set properties for landmarks
        world.landmarks[0].color = np.array([0.75, 0.25, 0.25])
        world.landmarks[1].color = np.array([0.25, 0.75, 0.25])
        world.landmarks[2].color = np.array([0.25, 0.25, 0.75])

        # Special colors for goals
        world.agents[0].goal_a.color = world.agents[0].goal_b.color                
        world.agents[1].goal_a.color = world.agents[1].goal_b.color

        # Set random initial states for agents
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dimension_position)
            agent.state.p_vel = np.zeros(world.dimension_position)
            agent.state.c = np.zeros(world.dimension_communication)

        #  Set random initial states for landmarks
        for landmark in world.landmarks:
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dimension_position)
            landmark.state.p_vel = np.zeros(world.dimension_position)

    def reward(self, agent, world):
        """
        Reward is collective. So agents have to learn to communicate the goal of the other agent,
        and navigate to their landmark.

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks

        Returns:
            (float) Negative distance squared of the other agent to their goal landmark
        """
        # Zero reward if agents has no goal
        if agent.goal_a is None or agent.goal_b is None:
            return 0.0

        # Negative distance squared of the other agent to their goal landmark
        dist2 = np.sum(np.square(agent.goal_a.state.p_pos - agent.goal_b.state.p_pos))
        return -dist2

    def observation(self, agent, world):
        """
        Define the observations.

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks

        Returns:
            (np.array) Observations array with the velocity of the agent,
                       distance to all landmarks in the agent's reference frame, goal color[1],
                       and the communication state of all other agents.
        """
        # Goal color
        goal_color = [np.zeros(world.dimension_color), np.zeros(world.dimension_color)]
        if agent.goal_b is not None:
            goal_color[1] = agent.goal_b.color 

        # Get positions and colors of all landmarks in this agent's reference frame
        landmarks_pos = []
        landmarks_color = []
        for landmark in world.landmarks:
            landmarks_pos.append(landmark.state.p_pos - agent.state.p_pos)
            landmarks_color.append(landmark.color)

        # Communication of all other agents
        comm = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)

        return np.concatenate([agent.state.p_vel] + landmarks_pos + [goal_color[1]] + comm)
